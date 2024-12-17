'use client'

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Languages } from "lucide-react"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"
import { SearchDialog } from "./search-dialog"

export function NavigationMenu({ className }: { className?: string }) {
    const pathname = usePathname()
    const { locale, toggleLocale } = useLocale()
    const t = i18n[locale]

    const links = [
        {
            href: '/',
            label: t.nav.home
        },
        {
            href: '/posts',
            label: t.nav.posts
        },
        {
            href: '/about',
            label: t.nav.about
        }
    ]

    return (
        <header className={className}>
            <div className="container flex h-14 items-center justify-between">
                <Link href="/" className="font-bold">
                    ZhuHai.Fun
                </Link>
                <nav className="flex items-center space-x-6">
                    <SearchDialog />
                    {links.map(({ href, label }) => (
                        <Link
                            key={href}
                            href={href}
                            className={cn(
                                "text-sm font-medium transition-colors hover:text-primary",
                                pathname === href
                                    ? "text-foreground"
                                    : "text-muted-foreground"
                            )}
                        >
                            {label}
                        </Link>
                    ))}
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={toggleLocale}
                        className="text-muted-foreground"
                    >
                        <Languages className="w-4 h-4 mr-2" />
                        {locale === 'zh' ? 'English' : '中文'}
                    </Button>
                </nav>
            </div>
        </header>
    )
} 