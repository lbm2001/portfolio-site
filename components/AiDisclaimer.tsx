// Shown on a project/post detail page when its `aiAssisted` field is set.
// Centralizing the wording here means it only needs updating in one place.
export default function AiDisclaimer() {
  return (
    <p className="ai-disclaimer">
      The ideas and work here are my own. An AI assistant (Claude Sonnet 5) helped with the write-up.
    </p>
  );
}
