import katex from "katex";
import Image from "next/image";

export interface Figure {
  id: string;
  src: string;
  alt: string;
  caption: string;
  width: number;
  height: number;
}

// Renders inline `$..$` math segments within a line of text, keeping the
// surrounding plain text untouched.
function renderInline(text: string): React.ReactNode[] {
  return text.split(/(\$[^$]+\$)/g).map((part, i) => {
    if (part.length > 2 && part.startsWith("$") && part.endsWith("$")) {
      const html = katex.renderToString(part.slice(1, -1), {
        throwOnError: false,
      });
      return (
        <span
          key={i}
          className="math-inline"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    }
    return part;
  });
}

// Renders a plain-text article body into paragraphs, with support for two
// block-level markers on their own blank-line-separated block:
//   $$ ... $$            -> a centered display equation (KaTeX, build-time)
//   {{figure:some-id}}   -> the Figure with that id from the `figures` list
// Everything else becomes a <p>, with inline `$..$` math rendered within it.
export function renderBody(body: string, figures?: Figure[]): React.ReactNode {
  return body
    .trim()
    .split(/\n\s*\n/)
    .map((block, i) => {
      const trimmed = block.trim();

      const figureMatch = trimmed.match(/^\{\{figure:([\w-]+)\}\}$/);
      if (figureMatch) {
        const fig = figures?.find((f) => f.id === figureMatch[1]);
        if (!fig) return null;
        return (
          <figure key={i} className="body-figure">
            <Image
              src={fig.src}
              alt={fig.alt}
              width={fig.width}
              height={fig.height}
              sizes="(max-width: 760px) 100vw, 680px"
            />
            <figcaption>{fig.caption}</figcaption>
          </figure>
        );
      }

      const mathMatch = trimmed.match(/^\$\$([\s\S]+)\$\$$/);
      if (mathMatch) {
        const html = katex.renderToString(mathMatch[1].trim(), {
          throwOnError: false,
          displayMode: true,
        });
        return (
          <div
            key={i}
            className="math-display"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }

      return <p key={i}>{renderInline(trimmed)}</p>;
    });
}
