import './globals.css'
import { Inter as FontSans } from 'next/font/google'
import { ThemeProvider } from "@/components/common/theme-provider"
import { NavigationMenu } from '@/components/common/navigation-menu'
import { cn } from "@/lib/utils"
import { Metadata } from "next"
import { siteConfig } from "@/config/site"
import Link from "next/link"
import { Analytics } from "@vercel/analytics/react"
import { SpeedInsights } from "@vercel/speed-insights/next"
import { LocaleProvider } from "@/contexts/locale-context"

export const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
})

export const metadata: Metadata = {
  title: {
    default: siteConfig.name,
    template: `%s | ${siteConfig.name}`,
  },
  description: siteConfig.description,
  keywords: siteConfig.keywords,
  authors: [{ name: siteConfig.author }],
  creator: siteConfig.author,
  openGraph: {
    type: "website",
    locale: "zh_CN",
    url: siteConfig.url,
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
    images: [
      {
        url: siteConfig.ogImage,
        width: 1200,
        height: 630,
        alt: siteConfig.name,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteConfig.name,
    description: siteConfig.description,
    images: [siteConfig.ogImage],
    creator: `@${siteConfig.author}`,
  },
  icons: {
    icon: "/icon.png",
    shortcut: "/favicon-16x16.png",
    apple: "/icon.png",
  },
  manifest: "/site.webmanifest",
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh" suppressHydrationWarning>
      <body className={cn("min-h-screen bg-background antialiased", fontSans.variable)}>
        <LocaleProvider>
          <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
            <div className="relative flex min-h-screen flex-col">
              <NavigationMenu className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60" />
              <main className="flex-1">
                {children}
                <Analytics />
                <SpeedInsights />
              </main>
              <footer className="border-t">
                <div className="container flex items-center justify-center h-14">
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span>友情链接:</span>
                    <Link
                      href="https://useai.cn"
                      target="_blank"
                      className="hover:text-primary transition-colors"
                    >
                      UseAI
                    </Link>
                    <span>•</span>
                    <Link
                      href="https://beian.miit.gov.cn/"
                      target="_blank"
                      className="hover:text-primary transition-colors"
                    >
                      豫ICP备2022008800号-1
                    </Link>
                  </div>
                </div>
              </footer>
            </div>
          </ThemeProvider>
        </LocaleProvider>
      </body>
    </html>
  )
}
