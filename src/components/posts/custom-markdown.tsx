'use client'

import { FC, memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypePrism from 'rehype-prism-plus'
import { CodeBlock } from './code-block'
import { useLocale } from '@/contexts/locale-context'

interface CustomMarkdownProps {
    children: string
}

export const CustomMarkdown: FC<CustomMarkdownProps> = memo(({ children }) => {
    const { locale } = useLocale()

    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypePrism]}
            components={{
                code({ className, children, ...props }) {
                    if (!className) {
                        return <code className="font-mono text-sm" {...props}>{children}</code>
                    }
                    return <CodeBlock className={className}>{children}</CodeBlock>
                },
                img: () => null // 暂时禁用图片渲染
            }}
            key={locale}
        >
            {children}
        </ReactMarkdown>
    )
})

CustomMarkdown.displayName = 'CustomMarkdown' 