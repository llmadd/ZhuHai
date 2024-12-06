'use client'

import { FC, memo } from 'react'
import ReactMarkdown, { Options, Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypePrism from 'rehype-prism-plus'
import slugify from 'slugify'

interface CustomMarkdownProps extends Options {
    className?: string
}

const MarkdownComponents: Components = {
    code: ({ className, children }) => {
        // 如果是行内代码（没有语言标识）
        if (!className) {
            return (
                <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-sm">
                    {children}
                </code>
            )
        }

        return (
            <pre className="not-prose">
                <code className={className}>{children}</code>
            </pre>
        )
    },

    h1: ({ children }) => (
        <h1 id={typeof children === 'string' ? slugify(children, { lower: true }) : undefined}
            className="scroll-m-20 text-4xl font-bold mb-4">
            {children}
        </h1>
    ),

    h2: ({ children }) => (
        <h2 id={typeof children === 'string' ? slugify(children, { lower: true }) : undefined}
            className="scroll-m-20 text-3xl font-semibold mb-3 mt-8">
            {children}
        </h2>
    ),

    h3: ({ children }) => (
        <h3 id={typeof children === 'string' ? slugify(children, { lower: true }) : undefined}
            className="scroll-m-20 text-2xl font-semibold mb-2 mt-6">
            {children}
        </h3>
    ),

    ul: ({ children }) => (
        <ul className="mb-2 list-disc pl-4 last:mb-0">{children}</ul>
    ),

    ol: ({ children }) => (
        <ol className="mb-2 list-decimal pl-4 last:mb-0">{children}</ol>
    ),

    li: ({ children }) => (
        <li className="mb-1 last:mb-0">{children}</li>
    ),

    blockquote: ({ children }) => (
        <blockquote className="mt-2 mb-2 border-l-4 border-border pl-4 italic">
            {children}
        </blockquote>
    ),
}

export const CustomMarkdown: FC<CustomMarkdownProps> = memo(
    ({ className, ...props }) => (
        <ReactMarkdown
            components={MarkdownComponents}
            className={className}
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypePrism]}
            {...props}
        />
    )
)

CustomMarkdown.displayName = 'CustomMarkdown' 