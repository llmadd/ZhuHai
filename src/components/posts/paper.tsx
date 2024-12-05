'use client'

import { Card } from "@/components/ui/card"
import Image from "next/image"
import Markdown, { Components } from 'react-markdown'
import slugify from 'slugify'

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

// 添加判断函数
const isExternalImage = (src: string) => {
    return src.startsWith('http://') || src.startsWith('https://')
}

const components: Partial<Components> = {
    h1: ({ children, ...props }) => (
        <h1 {...props} id={typeof children === 'string' ? slugify(children, { lower: true }) : undefined} className="scroll-m-20">
            {children}
        </h1>
    ),
    h2: ({ children, ...props }) => (
        <h2 {...props} id={typeof children === 'string' ? slugify(children, { lower: true }) : undefined} className="scroll-m-20">
            {children}
        </h2>
    ),
    h3: ({ children, ...props }) => (
        <h3 {...props} id={typeof children === 'string' ? slugify(children, { lower: true }) : undefined} className="scroll-m-20">
            {children}
        </h3>
    ),
}

export function Paper({ post }: PaperProps) {
    return (
        <Card className="p-6 max-w-4xl mx-auto">
            {post.coverImage && (
                <div className="relative w-full h-[300px] mb-6">
                    <Image
                        src={post.coverImage}
                        alt={post.title}
                        fill
                        className="object-cover rounded-lg"
                        {...(isExternalImage(post.coverImage) ? {
                            unoptimized: true,
                        } : {})}
                    />
                </div>
            )}

            <div className="prose prose-stone dark:prose-invert max-w-none">
                <Markdown components={components}>{post.content}</Markdown>
            </div>
        </Card>
    )
} 