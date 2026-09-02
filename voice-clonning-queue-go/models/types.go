package models

import "time"

// VoiceSpec mirrors the Python GPU service voice specification.
type VoiceSpec struct {
	Handle       *string `json:"handle,omitempty"`
	SpeakerID    *string `json:"speaker_id,omitempty"`
	RefText      *string `json:"ref_text,omitempty"`
	AllowSidecar bool    `json:"allow_sidecar"`
	Seed         bool    `json:"seed"`
}

// OutputSpec defines whether output is returned in-memory (npz/arrays) or files on disk.
type OutputSpec struct {
	Mode   string   `json:"mode"`
	JobDir *string  `json:"job_dir,omitempty"`
	Names  []string `json:"names,omitempty"`
}

// RenderRequest is the incoming payload for /v2/jobs/render.
type RenderRequest struct {
	JobID *string `json:"job_id,omitempty"`
	// ParentID ties this job to the upstream request it is one piece of. The
	// :8013 studio splits one take into one render job per emotion, so without
	// it the dashboard shows N unrelated rows for what the user asked once --
	// and an OOM on one piece leaves its siblings queued for a take that can no
	// longer be assembled.
	ParentID *string `json:"parent_id,omitempty"`
	// Request is the payload the *upstream* caller sent, verbatim -- the n8n body
	// the :8013 studio received, plus what it resolved that body into. The studio
	// rewrites voice_id, sex and donor_set with defaults before anything reaches
	// here, so without the original the dashboard cannot show that a fallback
	// happened. Opaque on purpose: the gateway only stores and serves it.
	Request   map[string]interface{} `json:"request,omitempty"`
	RawPrompt string                 `json:"raw_prompt,omitempty"`
	Prompt    string                 `json:"prompt,omitempty"`
	Chunks    []string               `json:"chunks"`
	Voice     *VoiceSpec             `json:"voice,omitempty"`
	CFGValue  float64                `json:"cfg_value"`
	Timesteps int                    `json:"timesteps"`
	LoRA      interface{}            `json:"lora,omitempty"`
	Output    OutputSpec             `json:"output"`
	Lane      string                 `json:"lane"`
	Client    string                 `json:"client"`
}

// JobStatus represents the state of a render job.
type JobStatus string

const (
	StatusQueued    JobStatus = "queued"
	StatusRunning   JobStatus = "running"
	StatusCompleted JobStatus = "completed"
	StatusFailed    JobStatus = "failed"
	StatusCancelled JobStatus = "cancelled"
)

// ErrorKind classifies a failure so the dashboard can say *why* without parsing
// a traceback, and so the queue knows which failures are worth cascading.
const (
	// ErrKindOOM is a CUDA out-of-memory failure. It is not the job's fault and
	// not retryable on the same GPU state, so the whole request is torn down.
	ErrKindOOM = "oom"
	// ErrKindUpstreamOOM marks a sibling cancelled because another piece of the
	// same request hit OOM -- the take can no longer be assembled, so finishing
	// this piece would burn GPU time for nothing.
	ErrKindUpstreamOOM = "upstream_oom"
)

// RenderJob represents an active or completed job in the queue.
type RenderJob struct {
	JobID    string                 `json:"job_id"`
	ParentID string                 `json:"parent_id,omitempty"`
	Request  map[string]interface{} `json:"request,omitempty"`
	// Engine is what the pipeline actually fed the model, PATCHed in once a take
	// has run: the donor clip each emotion was cloned from, the target SeedVC
	// converted into, and the knobs both were given. `request` says what was asked
	// for; this says what happened, and the two differ every time a default or a
	// random donor stands in.
	Engine      map[string]interface{} `json:"engine,omitempty"`
	RawPrompt   string                 `json:"raw_prompt,omitempty"`
	Chunks      []string               `json:"chunks"`
	Voice       *VoiceSpec             `json:"voice,omitempty"`
	CFGValue    float64                `json:"cfg_value"`
	Timesteps   int                    `json:"timesteps"`
	LoRA        interface{}            `json:"lora,omitempty"`
	Output      OutputSpec             `json:"output"`
	Lane        string                 `json:"lane"`
	Client      string                 `json:"client"`
	Status      JobStatus              `json:"status"`
	Position    *int                   `json:"position,omitempty"`
	Error       *string                `json:"error,omitempty"`
	ErrorKind   string                 `json:"error_kind,omitempty"`
	Result      map[string]interface{} `json:"result,omitempty"`
	Payload     []byte                 `json:"-"`
	AudioWAV    []byte                 `json:"-"`
	ChunksDone  int                    `json:"chunks_done"`
	TotalChunks int                    `json:"total_chunks"`
	Created     float64                `json:"created"`
	Started     *float64               `json:"started,omitempty"`
	Finished    *float64               `json:"finished,omitempty"`
	DoneChan    chan struct{}          `json:"-"`
	// External marks a visibility-only job the GPU worker never runs. A caller (the
	// :8013 studio) owns its whole lifecycle and drives status via PATCH, so it shows
	// up on the dashboard as one row per upstream request without competing for the GPU.
	External bool `json:"external,omitempty"`
}

// JobUpdate is the PATCH body for /v2/jobs/:job_id — every field optional, so a caller
// can advance status, publish the planned chunks, or attach a result independently.
type JobUpdate struct {
	Status      *string                 `json:"status,omitempty"`
	ChunksDone  *int                    `json:"chunks_done,omitempty"`
	TotalChunks *int                    `json:"total_chunks,omitempty"`
	Chunks      *[]string               `json:"chunks,omitempty"`
	Error       *string                 `json:"error,omitempty"`
	ErrorKind   *string                 `json:"error_kind,omitempty"`
	Result      *map[string]interface{} `json:"result,omitempty"`
	Engine      *map[string]interface{} `json:"engine,omitempty"`
}

// NewRenderJob creates an initialized RenderJob.
func NewRenderJob(req RenderRequest, jobID string) *RenderJob {
	now := float64(time.Now().UnixNano()) / 1e9
	cfg := req.CFGValue
	if cfg <= 0 {
		cfg = 2.0
	}
	steps := req.Timesteps
	if steps <= 0 {
		steps = 10
	}
	lane := req.Lane
	if lane != "interactive" && lane != "batch" {
		lane = "batch"
	}
	outMode := req.Output.Mode
	if outMode == "" {
		outMode = "npz"
	}
	output := req.Output
	output.Mode = outMode

	rawPrompt := req.RawPrompt
	if rawPrompt == "" {
		rawPrompt = req.Prompt
	}

	parentID := ""
	if req.ParentID != nil {
		parentID = *req.ParentID
	}

	return &RenderJob{
		JobID:       jobID,
		ParentID:    parentID,
		Request:     req.Request,
		RawPrompt:   rawPrompt,
		Chunks:      req.Chunks,
		Voice:       req.Voice,
		CFGValue:    cfg,
		Timesteps:   steps,
		LoRA:        req.LoRA,
		Output:      output,
		Lane:        lane,
		Client:      req.Client,
		Status:      StatusQueued,
		TotalChunks: len(req.Chunks),
		Created:     now,
		DoneChan:    make(chan struct{}),
	}
}

// AsDict converts a job into a JSON-friendly map with position.
func (j *RenderJob) AsDict(position *int) map[string]interface{} {
	now := float64(time.Now().UnixNano()) / 1e9
	waited := now - j.Created
	if j.Started != nil {
		waited = *j.Started - j.Created
	}

	var ran *float64
	if j.Started != nil {
		var r float64
		if j.Finished != nil {
			r = *j.Finished - *j.Started
		} else {
			r = now - *j.Started
		}
		ran = &r
	}

	// Three ways a row can offer playback, in the order GetAudio tries them: bytes
	// held in memory, a file still on this disk, or -- for an external row whose
	// owner already uploaded and deleted its scratch copy -- the delivered URL.
	hasAudio := len(j.AudioWAV) > 0
	if !hasAudio && j.Result != nil {
		if files, ok := j.Result["files"].([]interface{}); ok && len(files) > 0 {
			hasAudio = true
		}
		if url, ok := j.Result["file_url"].(string); ok && url != "" {
			hasAudio = true
		}
	}

	res := map[string]interface{}{
		"job_id":       j.JobID,
		"parent_id":    j.ParentID,
		"raw_prompt":   j.RawPrompt,
		"status":       j.Status,
		"chunks":       j.Chunks,
		"chunks_done":  j.ChunksDone,
		"total_chunks": j.TotalChunks,
		"voice":        j.Voice,
		"cfg_value":    j.CFGValue,
		"timesteps":    j.Timesteps,
		"lora":         j.LoRA,
		"output":       j.Output,
		"lane":         j.Lane,
		"client":       j.Client,
		"created":      j.Created,
		"started":      j.Started,
		"finished":     j.Finished,
		"waited_s":     waited,
		"ran_s":        ran,
		"has_audio":    hasAudio,
		"external":     j.External,
	}

	// Only when there is one: every render job carries this key otherwise, and an
	// empty object on the card reads as "the caller sent nothing".
	if len(j.Request) > 0 {
		res["request"] = j.Request
	}
	if len(j.Engine) > 0 {
		res["engine"] = j.Engine
	}

	if j.Status == StatusQueued && position != nil {
		res["position"] = *position
	}
	if j.Error != nil {
		res["error"] = *j.Error
	}
	if j.ErrorKind != "" {
		res["error_kind"] = j.ErrorKind
	}
	if j.Result != nil {
		res["result"] = j.Result
	}
	return res
}
