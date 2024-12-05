'use client'

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { useState } from "react"
import { Search } from "lucide-react"
import { useRouter } from "next/navigation"
import { Button } from "../ui/button"

interface SearchResult {
    title: string
    excerpt: string
    slug: string
}

export function SearchDialog() {
    const [open, setOpen] = useState(false)
    const [results, setResults] = useState<SearchResult[]>([])
    const [loading, setLoading] = useState(false)
    const router = useRouter()

    async function onSearch(query: string) {
        if (query.length < 2) {
            setResults([])
            return
        }

        setLoading(true)
        try {
            const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`)
            const data = await res.json()
            setResults(data)
        } catch (error) {
            console.error('Search failed:', error)
        } finally {
            setLoading(false)
        }
    }

    function onSelect(result: SearchResult) {
        setOpen(false)
        router.push(`/posts/${result.slug}`)
    }

    return (
        <>
            <Button
                variant="outline"
                size="icon"
                className="hidden sm:flex"
                onClick={() => setOpen(true)}
            >
                <Search className="h-4 w-4" />
            </Button>
            <Dialog open={open} onOpenChange={setOpen}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogTitle className="sr-only">搜索文章</DialogTitle>
                    <div className="space-y-4">
                        <Input
                            placeholder="搜索文章..."
                            onChange={(e) => onSearch(e.target.value)}
                        />
                        <div className="space-y-2">
                            {loading && <div className="text-center">搜索中...</div>}
                            {results.map((result) => (
                                <div
                                    key={result.slug}
                                    className="p-4 cursor-pointer hover:bg-muted rounded-lg"
                                    onClick={() => onSelect(result)}
                                >
                                    <h3 className="font-medium mb-1">{result.title}</h3>
                                    <p className="text-sm text-muted-foreground line-clamp-2">
                                        {result.excerpt}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                </DialogContent>
            </Dialog>
        </>
    )
} 