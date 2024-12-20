'use client'

import { Card } from "@/components/ui/card"
import Image from "next/image"
import { CustomMarkdown } from "./custom-markdown"
import { ShareCard } from "./share-card"
import { useState, useEffect } from "react"
import { useLocale } from "@/contexts/locale-context"
import { PostSidebar } from "./post-sidebar"
import { getTableOfContents } from "@/lib/toc"
import { PostHeader } from "@/components/posts/post-header"
import { Button } from "@/components/ui/button"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

interface PaperProps {
    post: {
        title: {
            zh: string
            en: string
        }
        date: string
        author: string
        content: {
            zh: string
            en: string
        }
        coverImage?: string
        coverImageAlt?: {
            zh: string
            en: string
        }
        tags?: string[]
        slug: string
    }
}

export function Paper({ post }: PaperProps) {
    const { locale } = useLocale()
    const [url, setUrl] = useState(`https://zhuhai.fun/posts/${post.slug}`)
    const [toc, setToc] = useState<any[]>([])
    const [showMobileToc, setShowMobileToc] = useState(false)

    useEffect(() => {
        setUrl(window.location.href)
        getTableOfContents(post.content[locale]).then(setToc)
    }, [post.content, locale])

    return (
        <>
            <PostHeader />
            <div className="container max-w-7xl mx-auto relative py-6 md:py-10">
                <div className="lg:hidden mb-4">
                    <Button
                        variant="outline"
                        className="w-full flex items-center justify-between"
                        onClick={() => setShowMobileToc(!showMobileToc)}
                    >
                        <span>目录</span>
                        <ChevronDown className={cn("w-4 h-4 transition-transform", showMobileToc && "rotate-180")} />
                    </Button>
                    {showMobileToc && (
                        <div className="mt-2 border rounded-lg p-4 bg-background">
                            <PostSidebar tableOfContents={toc} />
                        </div>
                    )}
                </div>

                <div className="flex flex-col lg:flex-row lg:gap-8">
                    <Card className="flex-1 p-4 lg:p-6 order-2 lg:order-1 lg:max-w-[calc(100%-18rem)]">
                        {post.coverImage && (
                            <div className="relative w-full mb-6">
                                <div className="relative w-full aspect-[16/9] overflow-hidden rounded-lg">
                                    <Image
                                        src={post.coverImage}
                                        alt={post.coverImageAlt?.[locale] || post.title[locale]}
                                        fill
                                        priority
                                        className="object-contain"
                                        unoptimized
                                    />
                                </div>
                            </div>
                        )}

                        <article className="prose prose-stone dark:prose-invert max-w-none">
                            <h1 className="mb-4 text-2xl lg:text-3xl font-bold">{post.title[locale]}</h1>
                            <CustomMarkdown>
                                {post.content[locale]}
                            </CustomMarkdown>
                        </article>

                        <ShareCard title={post.title[locale]} url={url} />
                    </Card>

                    <div className="hidden lg:block w-64 flex-shrink-0 order-1 lg:order-2">
                        <div className="sticky top-[calc(64px+3.5rem)] flex flex-col max-h-[calc(100vh-8rem-64px)]">
                            <PostSidebar tableOfContents={toc} />
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
} 