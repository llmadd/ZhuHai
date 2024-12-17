import { getAllPosts } from "@/lib/posts"
import { NextRequest, NextResponse } from "next/server"
import { Locale } from "@/config/i18n"

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams
    const query = searchParams.get('q')?.toLowerCase()
    const locale = (searchParams.get('locale') || 'zh') as Locale

    if (!query) {
        return NextResponse.json([])
    }

    const posts = await getAllPosts(false)
    const results = posts.filter(post => {
        const titleMatch = post.title[locale].toLowerCase().includes(query)
        const excerptMatch = post.excerpt[locale].toLowerCase().includes(query)
        const contentMatch = post.content[locale].toLowerCase().includes(query)
        return titleMatch || excerptMatch || contentMatch
    }).map(post => ({
        title: post.title[locale],
        excerpt: post.excerpt[locale],
        slug: post.slug
    }))

    return NextResponse.json(results.slice(0, 5))
} 