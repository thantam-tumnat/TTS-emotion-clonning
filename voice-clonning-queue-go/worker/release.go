package worker

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// Handing the card back when the queue drains.
//
// Both GPU services drop their weights on an idle timer -- three minutes by
// default. That timer has to be long, because neither service can tell "the
// batch is finished" from "the next take is two seconds away": the Go worker
// dispatches to /v2/direct_render, which bypasses the Python engine's own queue
// entirely, so `_queued()` over there is false even mid-batch. A short timer
// there would unload between every pair of jobs and buy a ~30 s reload for each.
//
// This gateway is the one process that knows. Telling the services outright
// turns three minutes of VRAM held for nobody into a couple of seconds, which is
// the whole point: another GPU application on this box gets the card back as
// soon as the render is genuinely over, rather than OOM-ing against weights that
// no longer have any work to do.
//
// The idle timers stay as the backstop for the paths this gateway cannot see.

const (
	defaultSeedVCURL = "http://127.0.0.1:8022"

	// Both endpoints wait for their service's GPU lock rather than interrupting a
	// running job, so this has to allow for a render this gateway did not
	// schedule finishing first -- while still bounding a wedged service to one
	// goroutine for 90 s instead of the lifetime of the process.
	releaseTimeout = 90 * time.Second

	// The pause before the call, re-checked against the queue afterwards. Jobs
	// often arrive back to back with a hairline gap between them, and the worker
	// loop is already asking for the next one; a beat of delay keeps a release
	// from racing a job that was submitted while we were deciding. It does not
	// pretend to cover an operator who re-renders half a minute later -- that
	// reload is the deliberate price of the card being free in between.
	defaultReleaseDelay = 2 * time.Second
)

// gpuReleaser asks the two GPU services to unload. Fire-and-forget by design: a
// release that fails changes nothing, because each service's idle timer is still
// there and still correct, just slower. Nothing in the render path waits on this
// or fails because of it.
type gpuReleaser struct {
	targets []string // full URLs, in the order they are called
	client  *http.Client
	delay   time.Duration
	q       interface{ Idle() bool }

	// One call in flight at a time. A second drain arriving while the first is
	// still waiting on a GPU lock would ask for the same thing again.
	mu sync.Mutex
}

func newGPUReleaser(pythonGPUURL, seedvcURL string, q interface{ Idle() bool }) *gpuReleaser {
	if seedvcURL == "" {
		seedvcURL = defaultSeedVCURL
	}
	return &gpuReleaser{
		// VoxCPM first: it is the larger holding by some margin (~5.6 GB against
		// ~2.3 GB), so if only one of the two calls gets through before the next
		// job arrives, this is the one worth having made.
		targets: []string{
			fmt.Sprintf("%s/v2/gpu/release", trimSlash(pythonGPUURL)),
			fmt.Sprintf("%s/release", trimSlash(seedvcURL)),
		},
		client: &http.Client{Timeout: releaseTimeout},
		delay:  releaseDelayFromEnv(),
		q:      q,
	}
}

// releaseDelayFromEnv reads GPU_RELEASE_DELAY, in seconds. "0" disables the
// pause (release the instant the queue drains); a negative or unparseable value
// falls back to the default rather than guessing at an intent.
func releaseDelayFromEnv() time.Duration {
	raw := os.Getenv("GPU_RELEASE_DELAY")
	if raw == "" {
		return defaultReleaseDelay
	}
	secs, err := strconv.ParseFloat(raw, 64)
	if err != nil || secs < 0 {
		fmt.Printf("[worker] ignoring GPU_RELEASE_DELAY=%q; using %s\n", raw, defaultReleaseDelay)
		return defaultReleaseDelay
	}
	return time.Duration(secs * float64(time.Second))
}

// Trigger runs a release in the background if one is not already running. Safe
// to call after every job: it returns immediately, and the queue is re-checked
// once the delay has elapsed, so a job that arrived in the meantime cancels it.
func (r *gpuReleaser) Trigger() {
	if !r.mu.TryLock() {
		return
	}
	go func() {
		defer r.mu.Unlock()

		if r.delay > 0 {
			time.Sleep(r.delay)
			if !r.q.Idle() {
				return // work arrived while we waited; the next drain will call again
			}
		}

		for _, url := range r.targets {
			r.call(url)
		}
	}()
}

func (r *gpuReleaser) call(url string) {
	req, err := http.NewRequest("POST", url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		// Not a job failure and not this gateway's problem to solve. A service
		// that is down is holding no VRAM to begin with.
		fmt.Printf("[worker] gpu release skipped (%s unreachable: %v)\n", url, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		fmt.Printf("[worker] gpu release refused by %s (HTTP %d)\n", url, resp.StatusCode)
		return
	}
	fmt.Printf("[worker] queue drained — asked %s to unload\n", url)
}

func trimSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
