'use client'

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Search } from "lucide-react"
import { useEffect, useState } from "react"
import { useDebounce } from "@/hooks/useDebounce"
import Link from "next/link"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

interface SearchResult {
    title: string
    excerpt: string
    slug: string
}

export function SearchDialog() {
    const [open, setOpen] = useState(false)
    const [query, setQuery] = useState("")
    const debouncedQuery = useDebounce(query, 300)
    const [results, setResults] = useState<SearchResult[]>([])
    const [isSearching, setIsSearching] = useState(false)
    const { locale } = useLocale()
    const t = i18n[locale]

    useEffect(() => {
        if (debouncedQuery.length > 0) {
            setIsSearching(true)
            fetch(`/api/search?q=${encodeURIComponent(debouncedQuery)}&locale=${locale}`)
                .then(res => res.json())
                .then(data => {
                    setResults(data)
                    setIsSearching(false)
                })
        } else {
            setResults([])
        }
    }, [debouncedQuery, locale])

    return (
        <>
            <button
                onClick={() => setOpen(true)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
                <Search className="w-4 h-4" />
                <span>{t.search.placeholder}</span>
                <kbd className="hidden md:inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100">
                    <span className="text-xs">⌘</span>K
                </kbd>
            </button>

            <Dialog open={open} onOpenChange={setOpen}>

                <DialogContent className="gap-0 p-0 outline-none">
                    <DialogTitle className="sr-only">{t.search.placeholder}</DialogTitle>
                    <div className="flex items-center border-b px-3">
                        <Search className="w-4 h-4 mr-2 shrink-0 opacity-50" />
                        <Input
                            className="h-12 border-0 focus-visible:ring-0"
                            placeholder={t.search.placeholder}
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                        />
                    </div>
                    <div className="max-h-[50vh] overflow-y-auto p-0">
                        {isSearching ? (
                            <p className="p-4 text-sm text-muted-foreground">
                                {t.search.searching}
                            </p>
                        ) : results.length > 0 ? (
                            <div className="grid gap-4 p-4">
                                {results.map((result) => (
                                    <Link
                                        key={result.slug}
                                        href={`/posts/${result.slug}`}
                                        onClick={() => setOpen(false)}
                                        className="space-y-1 rounded-lg border p-4 hover:bg-accent"
                                    >
                                        <h3 className="font-medium">{result.title}</h3>
                                        <p className="text-sm text-muted-foreground line-clamp-2">
                                            {result.excerpt}
                                        </p>
                                    </Link>
                                ))}
                            </div>
                        ) : query.length > 0 ? (
                            <p className="p-4 text-sm text-muted-foreground">
                                {t.search.noResults}
                            </p>
                        ) : null}
                    </div>
                </DialogContent>
            </Dialog>
        </>
    )
} 