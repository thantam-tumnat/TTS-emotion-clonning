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

	ok := q.Cancel("job_to_cancel")
	if !ok {
		t.Errorf("expected cancel to return true")
	}

	retrieved, _ := q.GetJob("job_to_cancel")
	if retrieved.Status != models.StatusCancelled {
		t.Errorf("expected status cancelled, got %s", retrieved.Status)
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
