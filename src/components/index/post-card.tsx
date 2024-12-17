'use client'

import { Card } from "@/components/ui/card"
import { Calendar, Folder, Tag, User } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { format } from "date-fns"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

// 判断是否为外部链接
const isExternalImage = (src: string) => {
    return src.startsWith('http://') || src.startsWith('https://')
}

interface PostCardProps {
    post: {
        slug: string
        title: {
            zh: string
            en: string
        }
        date: string
        author: string
        category: string
        excerpt: {
            zh: string
            en: string
        }
        coverImage?: string
        tags?: string[]
        status: 'published' | 'draft'
    }
}

export function PostCard({ post }: PostCardProps) {
    const { locale } = useLocale()
    const t = i18n[locale]

    return (
        <Card className="mb-6">
            <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6">
                <div className="flex-1 flex flex-col">
                    <Link
                        href={`/posts/${post.slug}`}
                        className="text-xl md:text-2xl font-bold hover:text-primary mb-3"
                    >
                        {post.title[locale]}
                    </Link>

                    <p className="text-muted-foreground mb-4 flex-1 line-clamp-2 md:line-clamp-3">
                        {post.excerpt[locale]}
                    </p>

                    <div className="flex flex-wrap items-center gap-3 md:gap-4 text-sm text-muted-foreground">
                        {post.author && (
                            <div className="flex items-center gap-1">
                                <User className="w-4 h-4" />
                                <span>{post.author}</span>
                            </div>
                        )}
                        <div className="flex items-center gap-1">
                            <Calendar className="w-4 h-4" />
                            <span>{format(new Date(post.date), locale === 'zh' ? 'yyyy年MM月dd日' : 'MMM dd, yyyy')}</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <Folder className="w-4 h-4" />
                            <span>{post.category}</span>
                        </div>
                        {post.tags && post.tags.length > 0 && (
                            <div className="flex items-center gap-2">
                                <Tag className="w-4 h-4" />
                                <div className="flex gap-1">
                                    {post.tags.map((tag, index) => (
                                        <span key={tag}>
                                            {tag}
                                            {index < post.tags!.length - 1 && ", "}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {post.coverImage && (
                    <div className="relative w-full md:w-[280px] flex-shrink-0 order-first md:order-last">
                        <div className="relative w-full aspect-[4/3] overflow-hidden rounded-lg">
                            <Image
                                src={post.coverImage}
                                alt={post.title[locale]}
                                fill
                                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                                className="object-contain"
                                unoptimized
                            />
                        </div>
                    </div>
                )}
            </div>
        </Card>
    )
} 