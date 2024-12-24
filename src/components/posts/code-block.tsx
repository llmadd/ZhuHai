'use client'

import { FC, useState, useRef } from 'react'
import { Check, Copy } from 'lucide-react'

interface CodeBlockProps {
    children: string
    className?: string
}

export const CodeBlock: FC<CodeBlockProps> = ({ children, className }) => {
    const [copied, setCopied] = useState(false)
    const preRef = useRef<HTMLPreElement>(null)

    const onCopy = async () => {
        try {
            await navigator.clipboard.writeText(children)
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        } catch (err) {
            console.error('Failed to copy text: ', err)
        }
    }

    return (
        <div className="relative">
            <pre className="not-prose !p-4 rounded-lg bg-muted overflow-x-auto" ref={preRef}>
                <code className={`${className} !whitespace-pre`}>{children}</code>
            </pre>
            <button
                className="absolute right-3 top-3 transition-colors p-2 rounded-md hover:bg-background"
                onClick={onCopy}
                title="复制代码"
            >
                {copied ? (
                    <Check className="h-4 w-4 text-green-500" />
                ) : (
                    <Copy className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                )}
            </button>
        </div>
    )
}