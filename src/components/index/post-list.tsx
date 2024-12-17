'use client'

import { PostCard } from "./post-card"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

interface Post {
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

interface Category {
    name: {
        zh: string
        en: string
    }
    key: string
}

interface PostListProps {
    posts: Post[]
    categories: Category[]
}

export function PostList({ posts, categories }: PostListProps) {
    const { locale } = useLocale()
    const t = i18n[locale]
    const [selectedCategory, setSelectedCategory] = useState<string>('all')

    const filteredPosts = selectedCategory === 'all'
        ? posts
        : posts.filter(post => post.category === selectedCategory)

    return (
        <section className="py-8 md:py-12">
            <div className="mb-8 md:mb-12">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-8">
                    <div>
                        <h1 className="text-3xl md:text-4xl font-bold mb-4">
                            {t.category.title}
                        </h1>
                        <p className="text-muted-foreground text-lg">
                            {t.category.description}
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <Button
                            key="all"
                            variant={selectedCategory === 'all' ? 'default' : 'outline'}
                            className="px-4"
                            onClick={() => setSelectedCategory('all')}
                        >
                            {t.category.allCategories}
                        </Button>
                        {categories.map((category) => (
                            <Button
                                key={category.key}
                                variant={selectedCategory === category.key ? 'default' : 'outline'}
                                className="px-4"
                                onClick={() => setSelectedCategory(category.key)}
                            >
                                {category.name[locale]}
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
                    {t.category.noPostsInCategory}
                </div>
            )}
        </section>
    )
} 