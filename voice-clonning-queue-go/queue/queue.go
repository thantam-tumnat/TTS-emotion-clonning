package queue

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"voice-cloning-queue/models"
)

const (
	MaxHistory = 500
)

// PriorityQueue manages strict FIFO job scheduling and job persistence in memory.
//
// There is one GPU behind this queue, so there is one waiting line: jobs run in the
// order they were submitted, whatever lane they declared. The lane is still recorded
// and reported — the dashboard splits its waiting counts by lane — but it no longer
// decides who goes first. A two-lane policy meant a `batch` job submitted seconds
// later could take the GPU ahead of an `interactive` job already waiting, which read
// from the dashboard as two jobs taking turns instead of one finishing first.
type PriorityQueue struct {
	mu      sync.RWMutex
	cond    *sync.Cond
	jobs    map[string]*models.RenderJob
	waiting []*models.RenderJob
	running *models.RenderJob
	closed  bool
}

// NewPriorityQueue creates an initialized PriorityQueue.
func NewPriorityQueue() *PriorityQueue {
	q := &PriorityQueue{
		jobs:    make(map[string]*models.RenderJob),
		waiting: make([]*models.RenderJob, 0),
	}
	q.cond = sync.NewCond(&q.mu)
	return q
}

// GenerateJobID creates a unique timestamped job identifier.
func (q *PriorityQueue) GenerateJobID() string {
	ts := time.Now().Format("20060102_150405")
	b := make([]byte, 4)
	rand.Read(b)
	return fmt.Sprintf("job_%s_%s", ts, hex.EncodeToString(b))
}

// Submit appends a job to the tail of the waiting line.
func (q *PriorityQueue) Submit(job *models.RenderJob) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.jobs[job.JobID] = job
	q.waiting = append(q.waiting, job)

	// Evict oldest finished jobs if over capacity
	if len(q.jobs) > MaxHistory {
		q.pruneOldJobs()
	}

	q.cond.Signal()
}

// NextJob blocks until a job is available, then returns the one that has waited
// longest. Strict FIFO — no lane may overtake another.
func (q *PriorityQueue) NextJob() *models.RenderJob {
	q.mu.Lock()
	defer q.mu.Unlock()

	for {
		if q.closed {
			return nil
		}

		if len(q.waiting) > 0 {
			job := q.waiting[0]
			q.waiting = q.waiting[1:]
			now := float64(time.Now().UnixNano()) / 1e9
			job.Status = models.StatusRunning
			job.Started = &now
			q.running = job
			return job
		}

		q.cond.Wait()
	}
}

// MarkCompleted updates a job state upon successful synthesis.
func (q *PriorityQueue) MarkCompleted(jobID string, result map[string]interface{}, payload []byte, audioWAV []byte) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return
	}

	// If job was cancelled while executing, preserve cancelled status
	if job.Status == models.StatusCancelled {
		if q.running != nil && q.running.JobID == jobID {
			q.running = nil
		}
		return
	}

	now := float64(time.Now().UnixNano()) / 1e9
	job.Status = models.StatusCompleted
	job.Finished = &now
	job.Result = result
	job.Payload = payload
	job.AudioWAV = audioWAV
	job.ChunksDone = job.TotalChunks

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	close(job.DoneChan)
}

// MarkFailed updates a job state upon failure.
func (q *PriorityQueue) MarkFailed(jobID string, errMsg string) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return
	}

	if job.Status == models.StatusCancelled {
		if q.running != nil && q.running.JobID == jobID {
			q.running = nil
		}
		return
	}

	now := float64(time.Now().UnixNano()) / 1e9
	job.Status = models.StatusFailed
	job.Finished = &now
	job.Error = &errMsg

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	close(job.DoneChan)
}

// Cancel marks a queued or running job as cancelled and removes it from the waiting line.
func (q *PriorityQueue) Cancel(jobID string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return false
	}

	if job.Status != models.StatusQueued && job.Status != models.StatusRunning {
		return false
	}

	job.Status = models.StatusCancelled
	now := float64(time.Now().UnixNano()) / 1e9
	job.Finished = &now
	errStr := "cancelled by user"
	job.Error = &errStr

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	for i, j := range q.waiting {
		if j.JobID == jobID {
			q.waiting = append(q.waiting[:i], q.waiting[i+1:]...)
			break
		}
	}

	close(job.DoneChan)
	return true
}

// GetJob retrieves a job and its queue position.
func (q *PriorityQueue) GetJob(jobID string) (*models.RenderJob, *int) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return nil, nil
	}

	if job.Status != models.StatusQueued {
		return job, nil
	}

	pos := q.calculatePosition(job)
	return job, &pos
}

// ListJobs returns a snapshot of all jobs.
func (q *PriorityQueue) ListJobs() []*models.RenderJob {
	q.mu.RLock()
	defer q.mu.RUnlock()

	res := make([]*models.RenderJob, 0, len(q.jobs))
	for _, j := range q.jobs {
		res = append(res, j)
	}
	return res
}

// GetPositions calculates current queue indices for all waiting jobs.
func (q *PriorityQueue) GetPositions() map[string]int {
	q.mu.RLock()
	defer q.mu.RUnlock()

	posMap := make(map[string]int)
	for pos, j := range q.waiting {
		posMap[j.JobID] = pos
	}
	return posMap
}

// GetStats returns summary counts of active, waiting, and completed jobs. The
// waiting figures stay split by lane so the dashboard cards keep working, but both
// lanes are now drained from the same line.
func (q *PriorityQueue) GetStats() (map[string]int, map[string]int, *models.RenderJob) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	counts := make(map[string]int)
	for _, j := range q.jobs {
		counts[string(j.Status)]++
	}

	waiting := map[string]int{"interactive": 0, "batch": 0}
	for _, j := range q.waiting {
		if j.Lane == "interactive" {
			waiting["interactive"]++
		} else {
			waiting["batch"]++
		}
	}

	return counts, waiting, q.running
}

// Wait blocks until a job finishes or timeout expires.
func (q *PriorityQueue) Wait(jobID string, timeout time.Duration) *models.RenderJob {
	q.mu.RLock()
	job, exists := q.jobs[jobID]
	q.mu.RUnlock()

	if !exists {
		return nil
	}

	select {
	case <-job.DoneChan:
		return job
	case <-time.After(timeout):
		return nil
	}
}

// Close signals the queue to stop waiting goroutines.
func (q *PriorityQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	q.cond.Broadcast()
}

func (q *PriorityQueue) calculatePosition(target *models.RenderJob) int {
	for pos, j := range q.waiting {
		if j.JobID == target.JobID {
			return pos
		}
	}
	return len(q.waiting)
}

func (q *PriorityQueue) pruneOldJobs() {
	var oldestID string
	var oldestTime float64 = 1e18

	for id, j := range q.jobs {
		if (j.Status == models.StatusCompleted || j.Status == models.StatusFailed || j.Status == models.StatusCancelled) && j.Created < oldestTime {
			oldestTime = j.Created
			oldestID = id
		}
	}

	if oldestID != "" {
		delete(q.jobs, oldestID)
	}
}
