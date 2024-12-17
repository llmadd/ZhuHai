'use client'

import { Button } from "@/components/ui/button"
import { ChevronLeft } from "lucide-react"
import Link from "next/link"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

export function PostHeader() {
    const { locale } = useLocale()
    const t = i18n[locale]

    return (
        <div className="sticky top-[64px] z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center">
                <Link href="/posts">
                    <Button variant="ghost" size="sm" className="gap-2">
                        <ChevronLeft className="h-4 w-4" />
                        {t.post.backToPostList}
                    </Button>
                </Link>
            </div>
        </div>
    )
} 