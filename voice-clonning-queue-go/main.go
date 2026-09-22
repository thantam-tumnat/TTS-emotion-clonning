package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"voice-cloning-queue/eventlog"
	"voice-cloning-queue/handlers"
	"voice-cloning-queue/queue"
	"voice-cloning-queue/worker"
)

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func main() {
	port := getEnv("PORT", "8020")
	pythonGPUURL := getEnv("PYTHON_GPU_URL", "http://127.0.0.1:8021")
	// Not a dispatch target: the worker calls this only to release VRAM once the
	// queue drains, so an unreachable SeedVC worker costs a log line, not a job.
	seedvcURL := getEnv("SEEDVC_URL", "http://127.0.0.1:8022")

	fmt.Println("================================================================")
	fmt.Println("🚀 SiangTTS High-Performance Go Fiber Queue Gateway (:8020)")
	fmt.Printf("👉 Python GPU Backend: %s\n", pythonGPUURL)
	fmt.Printf("👉 SeedVC worker (VRAM release only): %s\n", seedvcURL)
	fmt.Println("================================================================")

	// 1. Initialize Priority Queue & Worker
	logDir := getEnv("LOG_DIR", "logs")
	eventLogger, err := eventlog.New(logDir, "gateway")
	if err != nil {
		log.Fatalf("failed to initialize JSONL log directory %q: %v", logDir, err)
	}
	defer eventLogger.Close()
	_ = eventLogger.Write("info", "service_started", "", "Go Queue Gateway started", map[string]interface{}{
		"port": port, "python_gpu_url": pythonGPUURL,
	}, true)

	pq := queue.NewPriorityQueue(eventLogger)
	w := worker.NewWorker(pq, pythonGPUURL, seedvcURL)
	w.Start()
	defer w.Stop()
	defer pq.Close()

	// 2. Initialize Fiber App
	app := fiber.New(fiber.Config{
		AppName:               "SiangTTS Go Queue Gateway",
		BodyLimit:             50 * 1024 * 1024, // 50MB for reference audio uploads
		DisableStartupMessage: false,
	})

	// 3. Middlewares
	app.Use(recover.New())
	app.Use(logger.New(logger.Config{
		Format: "[${time}] ${status} - ${latency} ${method} ${path}\n",
	}))
	app.Use(func(c *fiber.Ctx) error {
		started := time.Now()
		err := c.Next()
		status := c.Response().StatusCode()
		// The dashboard polls these endpoints every second. Do not let healthy
		// polling drown out the useful job timeline; errors and all mutating
		// requests are still recorded.
		quietPoll := c.Method() == fiber.MethodGet && status < 400 &&
			(c.Path() == "/" || c.Path() == "/v2/jobs" || c.Path() == "/health" || c.Path() == "/api/logs")
		if !quietPoll {
			_ = eventLogger.Write("info", "http_request", "", c.Method()+" "+c.Path(), map[string]interface{}{
				"method": c.Method(), "path": c.Path(), "status": status,
				"duration_ms": time.Since(started).Milliseconds(),
			}, false)
		}
		return err
	})
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowHeaders: "*",
		AllowMethods: "GET,POST,HEAD,PUT,DELETE,PATCH,OPTIONS",
	}))

	// 4. Handlers
	jobsH := handlers.NewJobsHandler(pq)
	logsH := handlers.NewLogsHandler(eventLogger)
	proxyH := handlers.NewProxyHandler(pythonGPUURL)
	dashH := handlers.NewDashboardHandler(pq, pythonGPUURL)

	// 5. Job Queue Routes (Handled natively by Go)
	app.Post("/v2/jobs/render", jobsH.Render)
	app.Post("/v2/jobs/external", jobsH.SubmitExternal)
	app.Get("/v2/jobs", jobsH.List)
	app.Get("/v2/jobs/:job_id", jobsH.GetJob)
	app.Get("/v2/jobs/:job_id/result", jobsH.GetResult)
	app.Get("/v2/jobs/:job_id/audio", jobsH.GetAudio)
	app.Patch("/v2/jobs/:job_id", jobsH.UpdateJob)
	app.Delete("/v2/jobs/:job_id", jobsH.Cancel)
	app.Get("/api/logs", logsH.List)

	// 6. Voice, Speaker & Health Routes (Proxied to Python GPU :8021)
	app.Post("/v2/voices/resolve", proxyH.Forward)
	app.Post("/v2/voices", proxyH.Forward)
	app.Get("/v2/voices", proxyH.Forward)
	// Static paths first: Fiber matches in registration order, so ":handle"
	// and ":speaker_id" must not be allowed to swallow "seed".
	app.Post("/v2/voices/seed", proxyH.Forward)
	app.Delete("/v2/voices/seed", proxyH.Forward)
	app.Get("/v2/voices/:handle", proxyH.Forward)
	// Two segments -- ":handle" alone never matches this.
	app.Get("/v2/voices/:speaker_id/audio", proxyH.Forward)
	app.Delete("/v2/voices/:speaker_id", proxyH.Forward)
	app.Get("/health", proxyH.Forward)

	// 7. Web Dashboard UI
	app.Get("/", dashH.Index)

	// 8. Graceful Shutdown Setup
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Println("\n[shutdown] Gracefully stopping Go Queue Gateway...")
		_ = app.Shutdown()
	}()

	// 8. Start Listening
	addr := fmt.Sprintf("127.0.0.1:%s", port)
	log.Printf("[ready] Go Queue Gateway listening on http://%s\n", addr)
	if err := app.Listen(addr); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
