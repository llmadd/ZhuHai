'use client'

import { Card } from "@/components/ui/card"
import Image from "next/image"
import { CustomMarkdown } from "./custom-markdown"

interface PaperProps {
    post: {
        title: string
        date: string
        author: string
        content: string
        coverImage?: string
        tags?: string[]
    }
}

const isExternalImage = (src: string) => {
    return src.startsWith('http://') || src.startsWith('https://')
}

export function Paper({ post }: PaperProps) {
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
        </Card>
    )
} 