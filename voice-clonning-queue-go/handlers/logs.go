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
	page, err := eventlog.QueryPage(h.logger.Directory(), eventlog.QueryOptions{
		Date: c.Query("date"), Service: c.Query("service"), Level: c.Query("level"),
		JobID: c.Query("job_id"), Event: c.Query("event"), Query: c.Query("q"), Limit: limit,
		Before: c.Query("before"), After: c.Query("after"),
		ExcludeHTTP: c.Query("exclude_http") != "",
	})
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(fiber.Map{
		"logs": page.Records, "count": len(page.Records), "date": c.Query("date"), "service": c.Query("service"),
		"has_older": page.HasOlder, "has_newer": page.HasNewer,
	})
}
