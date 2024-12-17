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

    useEffect(() => {
        setUrl(window.location.href)
        getTableOfContents(post.content[locale]).then(setToc)
    }, [post.content, locale])

    return (
        <>
            <PostHeader />
            <div className="container relative py-6 md:py-10">
                <div className="flex flex-col lg:flex-row lg:gap-10">
                    <Card className="flex-1 p-6">
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
                            <h1 className="mb-4 text-3xl font-bold">{post.title[locale]}</h1>
                            <CustomMarkdown>
                                {post.content[locale]}
                            </CustomMarkdown>
                        </article>

                        <ShareCard title={post.title[locale]} url={url} />
                    </Card>

                    <div className="lg:w-64">
                        <PostSidebar tableOfContents={toc} />
                    </div>
                </div>
            </div>
        </>
    )
} 