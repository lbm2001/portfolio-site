import type { Metadata } from "next";
import { Archivo } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

// Self-hosted at build time by next/font — no runtime request to Google's
// servers (better privacy + no font-load flash). Exposed as the --font-archivo
// CSS variable that globals.css folds into --font-sans; Archivo is a variable
// font, so the whole 400–700 weight range comes in one file.
const archivo = Archivo({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-archivo",
});

export const metadata: Metadata = {
  title: "Lukas Müller · Robot Learning",
  description:
    "Making robots learn.",
};

// Set the theme BEFORE first paint to avoid a light-mode flash. Only an
// explicit user choice (stored by the nav toggle) is applied here as
// data-theme; with no stored choice the attribute stays off and the CSS
// `@media (prefers-color-scheme)` rule follows the OS. Rendered as the first
// child of <body> so it runs during HTML parse, ahead of any visible content
// (the App Router steers head tags through the Metadata API instead).
// suppressHydrationWarning covers the <html> attribute this mutates before
// React hydrates.
const noFlashTheme = `(function(){try{var t=localStorage.getItem('theme');if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t);}catch(e){}})();`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={archivo.variable} suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: noFlashTheme }} />
        {children}
      </body>
    </html>
  );
}
