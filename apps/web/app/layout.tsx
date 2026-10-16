import type { Metadata } from "next";
import Link from "next/link";
import ModelStatus from "@/components/ModelStatus";
import "./globals.css";

export const metadata: Metadata = {
  title: "Petti Vision",
  description: "Тэжээвэр амьтныг зургаар таних, Lost & Found хайлт",
};

const NAV = [
  { href: "/", label: "Таних" },
  { href: "/search", label: "Зургаар хайх" },
  { href: "/register", label: "Бүртгүүлэх" },
  { href: "/board", label: "Lost & Found" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="mn">
      <body>
        <ModelStatus />
        <header className="sticky top-0 z-20 border-b border-zinc-200/70 bg-[#fafafa]/80 backdrop-blur-md">
          <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4">
            <Link href="/" className="flex shrink-0 items-center gap-2 whitespace-nowrap text-[15px] font-semibold tracking-tight">
              <span className="h-2 w-2 rounded-full bg-brand-500" />
              Petti Vision
            </Link>
            <nav className="-mx-1 flex gap-0.5 overflow-x-auto text-sm">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href} className="whitespace-nowrap rounded-full px-3 py-1.5 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900">
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8 sm:py-12">{children}</main>
      </body>
    </html>
  );
}
