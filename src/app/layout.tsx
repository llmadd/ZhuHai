import './globals.css'
import { Inter as FontSans } from 'next/font/google'
import { ThemeProvider } from "@/components/common/theme-provider"
import { NavigationMenu } from '@/components/common/navigation-menu'
import { cn } from "@/lib/utils"

export const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh" suppressHydrationWarning>
      <body className={cn("min-h-screen bg-background antialiased", fontSans.variable)}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="relative flex min-h-screen flex-col">
            <NavigationMenu className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60" />
            <main className="flex-1">{children}</main>
            <footer className="border-t">
              <div className="container flex h-14 items-center">
                <p className="text-sm text-muted-foreground">
                  Built with Next.js and Tailwind CSS
                </p>
              </div>
            </footer>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
