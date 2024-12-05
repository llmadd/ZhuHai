import { getAllPosts } from "@/lib/posts"
import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams
    const query = searchParams.get('q')?.toLowerCase()

    if (!query) {
        return NextResponse.json([])
    }

    const posts = await getAllPosts(false)
    const results = posts.filter(post => {
        const titleMatch = post.title.toLowerCase().includes(query)
        const excerptMatch = post.excerpt.toLowerCase().includes(query)
        const contentMatch = post.content.toLowerCase().includes(query)
        return titleMatch || excerptMatch || contentMatch
    }).map(post => ({
        title: post.title,
        excerpt: post.excerpt,
        slug: post.slug
    }))

    return NextResponse.json(results.slice(0, 5))
} 