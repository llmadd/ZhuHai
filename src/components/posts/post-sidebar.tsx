'use client'

import { cn } from "@/lib/utils"
import { useEffect, useState } from "react"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

interface PostSidebarProps {
    tableOfContents: { id: string; text: string; level: number }[]
}

export function PostSidebar({ tableOfContents }: PostSidebarProps) {
    const { locale } = useLocale()
    const t = i18n[locale]
    const [activeId, setActiveId] = useState<string>("")

    const handleClick = (id: string, e: React.MouseEvent) => {
        e.preventDefault()
        let element = document.getElementById(id)

        if (!element) {
            const altId = id.replace(/^heading-/, '')
            element = document.getElementById(altId)
        }

        if (element) {
            setTimeout(() => {
                const offset = 80 // 顶部导航栏的高度
                const elementPosition = element!.getBoundingClientRect().top + window.scrollY
                const offsetPosition = elementPosition - offset

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                })
            }, 100)
        }
    }

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveId(entry.target.id)
                    }
                })
            },
            { rootMargin: '-80px 0px -35% 0px' }
        )

        const elements = document.querySelectorAll('[id]')
        elements.forEach((element) => {
            if (element.tagName.match(/^H[1-6]$/)) {
                observer.observe(element)
            }
        })

        return () => observer.disconnect()
    }, [])

    return (
        <div className="flex flex-col flex-1 min-h-0">
            <p className="font-medium text-sm mb-2 px-1 flex-shrink-0">{t.post.tableOfContents}</p>
            <nav className="flex-1 min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 scrollbar-track-transparent hover:scrollbar-thumb-gray-400 dark:hover:scrollbar-thumb-gray-500 pr-2">
                <div className="space-y-1">
                    {tableOfContents.map(({ id, text, level }) => (
                        <a
                            key={id}
                            href={`#${id}`}
                            onClick={(e) => handleClick(id, e)}
                            className={cn(
                                "block text-sm transition-colors hover:text-foreground py-1",
                                level === 2 ? "pl-3" : "pl-5",
                                activeId === id
                                    ? "text-foreground font-medium"
                                    : "text-muted-foreground"
                            )}
                        >
                            {text}
                        </a>
                    ))}
                </div>
            </nav>
        </div>
    )
} 