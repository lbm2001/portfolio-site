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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={archivo.variable}>
      <body>{children}</body>
    </html>
  );
}
