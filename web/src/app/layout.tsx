import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/header";
import { Providers } from "@/components/providers";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Assistant Portefeuille – multi-agent",
  description: "Assistant multi-agent de sélection de portefeuille : profil, régime de marché, allocation et explications.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fr" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Providers>
          <Header />
          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8">{children}</main>
          <footer className="border-t border-border py-6 text-center text-xs text-muted">
            Outil pédagogique – projet de Machine Learning, École Centrale de Casablanca. Les recommandations ne constituent
            pas un conseil en investissement. Actifs cotés en dollars US, hors effet de change.
          </footer>
        </Providers>
      </body>
    </html>
  );
}
