'use client'

import { PostCard } from "./post-card"
import { Button } from "@/components/ui/button"
import { useState } from "react"

interface PostListProps {
    posts: Array<{
        slug: string
        title: string
        date: string
        category: string
        excerpt: string
        coverImage?: string
        author?: string
        tags?: string[]
        status: 'published' | 'draft'
    }>
    categories: string[]
}

export function PostList({ posts, categories }: PostListProps) {
    const [selectedCategory, setSelectedCategory] = useState<string>('全部')

    const filteredPosts = selectedCategory === '全部'
        ? posts
        : posts.filter(post => post.category === selectedCategory)

    return (
        <section className="py-8 md:py-12">
            <div className="mb-8 md:mb-12">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-8">
                    <div>
                        <h1 className="text-3xl md:text-4xl font-bold mb-4">博客文章</h1>
                        <p className="text-muted-foreground text-lg">
                            分享技术文章和生活感悟
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <Button
                            key="全部"
                            variant={selectedCategory === '全部' ? 'default' : 'outline'}
                            className="px-4"
                            onClick={() => setSelectedCategory('全部')}
                        >
                            全部
                        </Button>
                        {categories.map((category) => (
                            <Button
                                key={category}
                                variant={selectedCategory === category ? 'default' : 'outline'}
                                className="px-4"
                                onClick={() => setSelectedCategory(category)}
                            >
                                {category}
                            </Button>
                        ))}
                    </div>
                </div>
            </div>

            <div className="space-y-6">
                {filteredPosts.map((post) => (
                    <PostCard key={post.slug} post={post} />
                ))}
            </div>

            {filteredPosts.length === 0 && (
                <div className="text-center text-muted-foreground py-12">
                    该分类下暂无文章
                </div>
            )}
        </section>
    )
} 