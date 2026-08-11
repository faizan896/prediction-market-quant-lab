import type { Metadata } from "next";

import { Nav } from "@/components/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuantLab",
  description:
    "Research → Backtest → Paper Trade. A calibration bench for prediction-market mispricing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="mx-auto max-w-[1440px] px-6">
          <Nav />
          <main className="pb-16">{children}</main>
          <footer className="border-t border-[var(--color-rule)] py-5">
            <p className="lab normal-case leading-relaxed tracking-[0.05em]">
              Research instrument. Nothing here places a real order, and none of it is financial
              advice. Prediction-market trading is restricted or prohibited in many jurisdictions —
              check yours.
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}
