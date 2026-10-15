import type { Metadata } from "next";
import Link from "next/link";
import ModelStatus from "@/components/ModelStatus";
import "./globals.css";

export const metadata: Metadata = {
  title: "Petti Vision",
  description: "Тэжээвэр амьтныг зургаар таних, Lost & Found хайлт",
};

const NAV = [
  { href: "/", label: "Зураг шинжлэх" },
  { href: "/register", label: "Бүртгүүлэх" },
  { href: "/board", label: "Lost & Found" },
  { href: "/search", label: "Зургаар хайх" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="mn">
      <body>
        <ModelStatus />
        <header className="sticky top-0 z-10 border-b border-stone-200 bg-white/90 backdrop-blur">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
            <Link href="/" className="text-lg font-bold tracking-tight">
              <span className="text-brand-600">Petti</span> Vision
            </Link>
            <nav className="flex flex-wrap gap-1 text-sm">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href} className="rounded-lg px-3 py-1.5 font-medium text-stone-600 hover:bg-stone-100 hover:text-stone-900">
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
