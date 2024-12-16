'use client'

import { Card } from "@/components/ui/card"
import Image from "next/image"
import { CustomMarkdown } from "./custom-markdown"
import { ShareCard } from "./share-card"
import { useEffect, useState } from "react"

interface PaperProps {
    post: {
        title: string
        date: string
        author: string
        content: string
        coverImage?: string
        tags?: string[]
        slug: string
    }
}

const isExternalImage = (src: string) => {
    return src.startsWith('http://') || src.startsWith('https://')
}

export function Paper({ post }: PaperProps) {
    const [url, setUrl] = useState(`https://zhuhai.fun/posts/${post.slug}`)

    useEffect(() => {
        // 在客户端更新 URL
        setUrl(window.location.href)
    }, [])

    return (
        <Card className="p-6 max-w-4xl mx-auto">
            {post.coverImage && (
                <div className="relative w-full mb-6">
                    <div className="relative w-full aspect-[16/9] overflow-hidden rounded-lg">
                        <Image
                            src={post.coverImage}
                            alt={post.title}
                            fill
                            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                            className="object-contain"
                            {...(isExternalImage(post.coverImage) ? {
                                unoptimized: true,
                            } : {})}
                        />
                    </div>
                </div>
            )}

            <div className="prose prose-stone dark:prose-invert max-w-none [&_img]:rounded-lg [&_img]:w-full [&_img]:h-auto">
                <CustomMarkdown>{post.content}</CustomMarkdown>
            </div>

            <ShareCard title={post.title} url={url} />
        </Card>
    )
} 