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

// SubmitExternal registers a visibility-only job that the GPU worker never runs. It
// lands in the jobs map (so the dashboard lists it) but not in the waiting line, and
// the worker is not signalled — the owner drives its status through UpdateJob instead.
func (q *PriorityQueue) SubmitExternal(job *models.RenderJob) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job.External = true
	if job.Status == "" {
		job.Status = models.StatusQueued
	}
	q.jobs[job.JobID] = job

	if len(q.jobs) > MaxHistory {
		q.pruneOldJobs()
	}
}

// UpdateJob applies an external status/progress update (PATCH). It is the only way an
// External job advances: queued -> running stamps Started, and any terminal status
// stamps Finished and releases anyone waiting on DoneChan. Returns false for an
// unknown job.
func (q *PriorityQueue) UpdateJob(jobID string, upd models.JobUpdate) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return false
	}

	now := float64(time.Now().UnixNano()) / 1e9

	if upd.Chunks != nil {
		job.Chunks = *upd.Chunks
		job.TotalChunks = len(*upd.Chunks)
	}
	if upd.TotalChunks != nil {
		job.TotalChunks = *upd.TotalChunks
	}
	if upd.ChunksDone != nil {
		job.ChunksDone = *upd.ChunksDone
	}
	if upd.Result != nil {
		job.Result = *upd.Result
	}
	if upd.Engine != nil {
		job.Engine = *upd.Engine
	}
	if upd.Error != nil {
		job.Error = upd.Error
	}
	if upd.ErrorKind != nil {
		job.ErrorKind = *upd.ErrorKind
	}

	if upd.Status != nil {
		st := models.JobStatus(*upd.Status)
		switch st {
		case models.StatusRunning:
			if job.Started == nil {
				job.Started = &now
			}
			job.Status = st
		case models.StatusCompleted, models.StatusFailed, models.StatusCancelled:
			job.Status = st
			job.Finished = &now
			if st == models.StatusCompleted {
				job.ChunksDone = job.TotalChunks
			}
			// Release waiters exactly once, even on a repeated terminal PATCH.
			select {
			case <-job.DoneChan:
			default:
				close(job.DoneChan)
			}
		default:
			job.Status = st
		}
	}

	return true
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

// MarkFailed updates a job state upon failure. `kind` classifies it (see
// models.ErrKind*); pass "" when the failure has no special meaning.
//
// A kind of OOM tears down the whole request, not just this piece: the studio
// splits one take into one job per emotion, and a take missing an emotion cannot
// be assembled. Leaving the siblings queued would spend GPU time -- on the very
// GPU that just ran out of memory -- producing audio nobody can use. Returns the
// ids that were cancelled as collateral, for the caller to log.
func (q *PriorityQueue) MarkFailed(jobID string, errMsg string, kind string) []string {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return nil
	}

	if job.Status == models.StatusCancelled {
		if q.running != nil && q.running.JobID == jobID {
			q.running = nil
		}
		return nil
	}

	now := float64(time.Now().UnixNano()) / 1e9
	job.Status = models.StatusFailed
	job.Finished = &now
	job.Error = &errMsg
	job.ErrorKind = kind

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	close(job.DoneChan)

	if kind == models.ErrKindOOM {
		return q.cancelGroupLocked(job, models.ErrKindUpstreamOOM,
			"cancelled: another chunk of this request hit CUDA OOM")
	}
	return nil
}

// GroupID is the request a job belongs to: its parent when it is one piece of a
// larger take, otherwise itself. Every job therefore has a group, so the
// dashboard can render one card per request without special-casing.
func GroupID(j *models.RenderJob) string {
	if j.ParentID != "" {
		return j.ParentID
	}
	return j.JobID
}

// cancelGroupLocked cancels the active members of `origin`'s group, skipping
// `origin` itself (it already has its own terminal state). Caller holds q.mu.
func (q *PriorityQueue) cancelGroupLocked(origin *models.RenderJob, kind string, reason string) []string {
	group := GroupID(origin)
	now := float64(time.Now().UnixNano()) / 1e9
	cancelled := make([]string, 0, 4)

	for _, sib := range q.jobs {
		if sib.JobID == origin.JobID || GroupID(sib) != group {
			continue
		}
		if sib.Status != models.StatusQueued && sib.Status != models.StatusRunning {
			continue
		}
		sib.Status = models.StatusCancelled
		sib.Finished = &now
		msg := reason
		sib.Error = &msg
		sib.ErrorKind = kind
		if q.running != nil && q.running.JobID == sib.JobID {
			q.running = nil
		}
		q.removeFromWaitingLocked(sib.JobID)
		select {
		case <-sib.DoneChan:
		default:
			close(sib.DoneChan)
		}
		cancelled = append(cancelled, sib.JobID)
	}
	return cancelled
}

func (q *PriorityQueue) removeFromWaitingLocked(jobID string) {
	for i, j := range q.waiting {
		if j.JobID == jobID {
			q.waiting = append(q.waiting[:i], q.waiting[i+1:]...)
			return
		}
	}
}

// Cancel marks a queued or running job as cancelled and removes it from the waiting
// line, along with the rest of its request group. Returns whether the named job was
// cancelled and the ids of the siblings that went with it.
func (q *PriorityQueue) Cancel(jobID string) (bool, []string) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return false, nil
	}

	if job.Status != models.StatusQueued && job.Status != models.StatusRunning {
		return false, nil
	}

	job.Status = models.StatusCancelled
	now := float64(time.Now().UnixNano()) / 1e9
	job.Finished = &now
	errStr := "cancelled by user"
	job.Error = &errStr

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	q.removeFromWaitingLocked(jobID)

	close(job.DoneChan)

	// One click, one take: cancelling any piece abandons the request it belongs
	// to. A half-cancelled take is not a useful state -- the caller cannot
	// assemble it, and the remaining pieces would still occupy the GPU.
	return true, q.cancelGroupLocked(job, "", "cancelled with the rest of this request")
}

// GetJob retrieves a job and its queue position.
func (q *PriorityQueue) GetJob(jobID string) (*models.RenderJob, *int) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return nil, nil
	}

	// External jobs are not in the GPU waiting line, so a queue position is meaningless.
	if job.Status != models.StatusQueued || job.External {
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

// Idle reports that the GPU has nothing left to do for this gateway: nothing
// running, nothing waiting, and no external job still in flight.
//
// The worker asks after every job, to decide whether the GPU services may hand
// their VRAM back. External jobs are included even though the worker never runs
// them: they are this gateway's only visibility into GPU work driven by someone
// else (a studio render straight to :8021), and telling the services to unload
// underneath one would buy a reload for memory nobody gained.
func (q *PriorityQueue) Idle() bool {
	q.mu.RLock()
	defer q.mu.RUnlock()

	if q.running != nil || len(q.waiting) > 0 {
		return false
	}
	for _, j := range q.jobs {
		if j.External && (j.Status == models.StatusQueued || j.Status == models.StatusRunning) {
			return false
		}
	}
	return true
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

// pruneOldJobs drops finished jobs until the map is back under MaxHistory.
//
// It loops: one call used to delete a single job, so a burst that pushed the map
// several jobs over the cap left it over the cap, and the retained audio (which
// is the bulk of a job's memory) stayed with it.
func (q *PriorityQueue) pruneOldJobs() {
	for len(q.jobs) > MaxHistory {
		var oldestID string
		var oldestTime float64 = 1e18

		for id, j := range q.jobs {
			if (j.Status == models.StatusCompleted || j.Status == models.StatusFailed || j.Status == models.StatusCancelled) && j.Created < oldestTime {
				oldestTime = j.Created
				oldestID = id
			}
		}

		if oldestID == "" {
			return // nothing terminal left to drop; the rest are queued or running
		}
		delete(q.jobs, oldestID)
	}
}
