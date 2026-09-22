package handlers

import (
	"strconv"

	"github.com/gofiber/fiber/v2"
	"voice-cloning-queue/eventlog"
)

type LogsHandler struct{ logger *eventlog.Logger }

func NewLogsHandler(logger *eventlog.Logger) *LogsHandler { return &LogsHandler{logger: logger} }

// List returns the newest matching JSONL records. The dashboard can poll this
// endpoint without knowing anything about files or the on-disk layout.
func (h *LogsHandler) List(c *fiber.Ctx) error {
	limit, _ := strconv.Atoi(c.Query("limit", "100"))
	entries, err := eventlog.Query(h.logger.Directory(), eventlog.QueryOptions{
		Date: c.Query("date"), Service: c.Query("service"), Level: c.Query("level"),
		JobID: c.Query("job_id"), Event: c.Query("event"), Query: c.Query("q"), Limit: limit,
	})
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(fiber.Map{"logs": entries, "count": len(entries), "date": c.Query("date"), "service": c.Query("service")})
}
