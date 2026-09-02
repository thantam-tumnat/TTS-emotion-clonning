package queue

import (
	"testing"
	"time"

	"voice-cloning-queue/models"
)

func TestPriorityQueue_BasicSubmitAndNext(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	req := models.RenderRequest{
		Chunks: []string{"test 1", "test 2"},
		Lane:   "batch",
	}
	job := models.NewRenderJob(req, "job_1")
	q.Submit(job)

	retrieved, pos := q.GetJob("job_1")
	if retrieved == nil {
		t.Fatalf("expected job_1 to exist")
	}
	if *pos != 0 {
		t.Errorf("expected position 0, got %d", *pos)
	}

	next := q.NextJob()
	if next == nil || next.JobID != "job_1" {
		t.Fatalf("expected next job to be job_1")
	}
	if next.Status != models.StatusRunning {
		t.Errorf("expected status running, got %s", next.Status)
	}

	q.MarkCompleted("job_1", map[string]interface{}{"status": "ok"}, nil, nil)

	retrieved, _ = q.GetJob("job_1")
	if retrieved.Status != models.StatusCompleted {
		t.Errorf("expected status completed, got %s", retrieved.Status)
	}
}

func TestPriorityQueue_StrictFIFO(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	// A later `interactive` job must not overtake `batch` jobs already waiting:
	// one GPU, one line, submission order decides.
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"b1"}, Lane: "batch"}, "batch_1"))
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"b2"}, Lane: "batch"}, "batch_2"))
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"i1"}, Lane: "interactive"}, "interactive_1"))

	for _, want := range []string{"batch_1", "batch_2", "interactive_1"} {
		got := q.NextJob()
		if got.JobID != want {
			t.Fatalf("expected %s, got %s", want, got.JobID)
		}
	}
}

func TestPriorityQueue_BatchNeverJumpsWaitingInteractive(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	// The regression: four interactive jobs queued, then one batch job. Under the
	// old burst policy the batch job ran fourth, ahead of an interactive job that
	// had already been waiting.
	for _, id := range []string{"i1", "i2", "i3", "i4"} {
		q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{id}, Lane: "interactive"}, id))
	}
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"b1"}, Lane: "batch"}, "b1"))

	for _, want := range []string{"i1", "i2", "i3", "i4", "b1"} {
		got := q.NextJob()
		if got.JobID != want {
			t.Fatalf("expected %s, got %s", want, got.JobID)
		}
	}
}

func TestPriorityQueue_CancelJob(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	job := models.NewRenderJob(models.RenderRequest{Chunks: []string{"test"}, Lane: "batch"}, "job_to_cancel")
	q.Submit(job)

	ok, _ := q.Cancel("job_to_cancel")
	if !ok {
		t.Errorf("expected cancel to return true")
	}

	retrieved, _ := q.GetJob("job_to_cancel")
	if retrieved.Status != models.StatusCancelled {
		t.Errorf("expected status cancelled, got %s", retrieved.Status)
	}
}

// submitGroup queues `n` sibling render jobs that all belong to one request.
func submitGroup(q *PriorityQueue, parent string, ids ...string) {
	for _, id := range ids {
		q.Submit(models.NewRenderJob(models.RenderRequest{
			Chunks:   []string{"test"},
			Lane:     "batch",
			ParentID: &parent,
		}, id))
	}
}

// An OOM on one emotion of a take must abandon the whole take: the remaining
// pieces cannot be assembled into anything, and running them would spend the very
// GPU memory that just ran out.
func TestMarkFailedOOM_CancelsSiblings(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	submitGroup(q, "req_1", "g_angry", "g_happy", "g_sad")
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"x"}, Lane: "batch"}, "g_unrelated"))

	collateral := q.MarkFailed("g_angry", "CUDA out of memory", models.ErrKindOOM)
	if len(collateral) != 2 {
		t.Fatalf("expected 2 siblings cancelled, got %d (%v)", len(collateral), collateral)
	}

	for _, id := range []string{"g_happy", "g_sad"} {
		j, _ := q.GetJob(id)
		if j.Status != models.StatusCancelled {
			t.Errorf("%s: expected cancelled, got %s", id, j.Status)
		}
		if j.ErrorKind != models.ErrKindUpstreamOOM {
			t.Errorf("%s: expected error_kind %q, got %q", id, models.ErrKindUpstreamOOM, j.ErrorKind)
		}
	}

	failed, _ := q.GetJob("g_angry")
	if failed.Status != models.StatusFailed || failed.ErrorKind != models.ErrKindOOM {
		t.Errorf("origin job: expected failed/oom, got %s/%s", failed.Status, failed.ErrorKind)
	}

	other, _ := q.GetJob("g_unrelated")
	if other.Status != models.StatusQueued {
		t.Errorf("a job outside the request group must be untouched, got %s", other.Status)
	}
}

// A non-OOM failure is this job's problem alone -- an empty chunk list or a bad
// voice handle says nothing about whether its siblings can run.
func TestMarkFailedOther_LeavesSiblings(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	submitGroup(q, "req_2", "g_a", "g_b")
	if collateral := q.MarkFailed("g_a", "unknown voice: nope", ""); len(collateral) != 0 {
		t.Fatalf("expected no cascade for a plain failure, got %v", collateral)
	}
	if j, _ := q.GetJob("g_b"); j.Status != models.StatusQueued {
		t.Errorf("expected sibling still queued, got %s", j.Status)
	}
}

// Cancelling one piece from the dashboard abandons the request it belongs to.
func TestCancel_CascadesToRequestGroup(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	submitGroup(q, "req_3", "g_x", "g_y", "g_z")
	ok, also := q.Cancel("g_x")
	if !ok || len(also) != 2 {
		t.Fatalf("expected cancel of the whole group, got ok=%v also=%v", ok, also)
	}
	if len(q.waiting) != 0 {
		t.Errorf("expected the waiting line drained, %d left", len(q.waiting))
	}
}

// pruneOldJobs used to delete one job per call, so a burst left the map over cap.
func TestPruneOldJobs_DrainsToCap(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	for i := 0; i < MaxHistory+25; i++ {
		id := "j" + string(rune('a'+i%26)) + string(rune(i))
		job := models.NewRenderJob(models.RenderRequest{Chunks: []string{"x"}, Lane: "batch"}, id)
		q.Submit(job)
		q.MarkCompleted(id, map[string]interface{}{"mode": "npz"}, nil, []byte{1, 2, 3})
	}
	if len(q.jobs) > MaxHistory {
		t.Errorf("expected history drained to %d, got %d", MaxHistory, len(q.jobs))
	}
}

func TestPriorityQueue_WaitTimeout(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	job := models.NewRenderJob(models.RenderRequest{Chunks: []string{"test"}, Lane: "batch"}, "job_wait")
	q.Submit(job)

	// Wait with 50ms timeout (should return nil because job is still queued)
	finished := q.Wait("job_wait", 50*time.Millisecond)
	if finished != nil {
		t.Errorf("expected nil on timeout, got %v", finished)
	}

	// Mark completed
	go func() {
		time.Sleep(20 * time.Millisecond)
		q.MarkCompleted("job_wait", map[string]interface{}{"status": "ok"}, nil, nil)
	}()

	finished = q.Wait("job_wait", 100*time.Millisecond)
	if finished == nil || finished.Status != models.StatusCompleted {
		t.Errorf("expected completed job, got %v", finished)
	}
}

func TestPriorityQueue_RawPrompt(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	raw := "(tone: sad) สวัสดีครับวันนี้เหนื่อยจัง (tone: happy) แต่พรุ่งนี้จะได้พักแล้ว!"
	req := models.RenderRequest{
		RawPrompt: raw,
		Chunks:    []string{"สวัสดีครับวันนี้เหนื่อยจัง", "แต่พรุ่งนี้จะได้พักแล้ว!"},
		Lane:      "interactive",
		Client:    "voxcpm-vc",
	}
	job := models.NewRenderJob(req, "job_raw_1")
	if job.RawPrompt != raw {
		t.Errorf("expected raw_prompt %q, got %q", raw, job.RawPrompt)
	}

	dict := job.AsDict(nil)
	if dict["raw_prompt"] != raw {
		t.Errorf("expected as_dict raw_prompt %q, got %v", raw, dict["raw_prompt"])
	}
}

func TestPriorityQueue_ExternalJobLifecycle(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	req := models.RenderRequest{
		JobID:     strPtr("meta_1"),
		RawPrompt: "สวัสดีครับ",
		Lane:      "batch",
		Client:    "voxcpm-vc",
	}
	job := models.NewRenderJob(req, "meta_1")
	q.SubmitExternal(job)

	// Visible in the map, but NOT in the GPU waiting line: NextJob must not see it.
	retrieved, pos := q.GetJob("meta_1")
	if retrieved == nil {
		t.Fatalf("expected meta_1 to exist")
	}
	if !retrieved.External {
		t.Errorf("expected External=true")
	}
	if pos != nil {
		t.Errorf("external job should have no queue position, got %d", *pos)
	}
	if len(q.waiting) != 0 {
		t.Fatalf("external job must not enter the waiting line, got %d waiting", len(q.waiting))
	}

	// PATCH: publish chunks, then advance running -> completed.
	chunks := []string{"สวัสดี", "ครับ"}
	if !q.UpdateJob("meta_1", models.JobUpdate{Chunks: &chunks}) {
		t.Fatalf("UpdateJob(chunks) returned false")
	}
	retrieved, _ = q.GetJob("meta_1")
	if retrieved.TotalChunks != 2 {
		t.Errorf("expected TotalChunks 2, got %d", retrieved.TotalChunks)
	}

	running := "running"
	q.UpdateJob("meta_1", models.JobUpdate{Status: &running})
	retrieved, _ = q.GetJob("meta_1")
	if retrieved.Status != models.StatusRunning || retrieved.Started == nil {
		t.Errorf("expected running with Started stamped, got %s started=%v", retrieved.Status, retrieved.Started)
	}

	completed := "completed"
	result := map[string]interface{}{"file_url": "https://x/y.wav"}
	q.UpdateJob("meta_1", models.JobUpdate{Status: &completed, Result: &result})
	retrieved, _ = q.GetJob("meta_1")
	if retrieved.Status != models.StatusCompleted || retrieved.Finished == nil {
		t.Errorf("expected completed with Finished stamped")
	}
	if retrieved.ChunksDone != retrieved.TotalChunks {
		t.Errorf("expected ChunksDone == TotalChunks on completion")
	}
	if retrieved.Result["file_url"] != "https://x/y.wav" {
		t.Errorf("expected file_url in result, got %v", retrieved.Result["file_url"])
	}

	// DoneChan must be closed exactly once — a repeated terminal PATCH must not panic.
	q.UpdateJob("meta_1", models.JobUpdate{Status: &completed})

	if q.UpdateJob("nope", models.JobUpdate{Status: &running}) {
		t.Errorf("UpdateJob on unknown id should return false")
	}
}

func strPtr(s string) *string { return &s }
