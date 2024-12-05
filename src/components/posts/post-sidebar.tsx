'use client'

import { cn } from "@/lib/utils"
import { useEffect, useState } from "react"

interface PostSidebarProps {
    tableOfContents: { id: string; text: string; level: number }[]
}

export function PostSidebar({ tableOfContents }: PostSidebarProps) {
    const [activeId, setActiveId] = useState<string>("")

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveId(entry.target.id)
                    }
                })
            },
            { rootMargin: '-20% 0px -35% 0px' }
        )

        const elements = document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]')
        elements.forEach((element) => observer.observe(element))

        return () => observer.disconnect()
    }, [])

    return (
        <div className="hidden lg:block">
            <div className="lg:fixed lg:top-[calc(64px+5rem)] lg:right-[max(2rem,calc(50%-45rem))] w-64 overflow-auto max-h-[calc(100vh-10rem)]">
                <div className="space-y-2 pb-8">
                    <p className="font-medium">目录</p>
                    <div className="space-y-1">
                        {tableOfContents.map(({ id, text, level }) => (
                            <a
                                key={`toc-${id}`}
                                href={`#${id}`}
                                className={cn(
                                    "block text-sm transition-colors hover:text-foreground",
                                    level === 2 ? "pl-4" : "pl-8",
                                    activeId === id
                                        ? "text-foreground font-medium"
                                        : "text-muted-foreground"
                                )}
                            >
                                {text}
                            </a>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
} 