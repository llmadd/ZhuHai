'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import Cookies from 'js-cookie'

type Locale = 'zh' | 'en'

interface LocaleContextType {
    locale: Locale
    toggleLocale: () => void
}

const LocaleContext = createContext<LocaleContextType | undefined>(undefined)

export function LocaleProvider({ children }: { children: ReactNode }) {
    const [locale, setLocale] = useState<Locale>('zh')

    useEffect(() => {
        const savedLocale = (Cookies.get('locale') as Locale) || 'zh'
        setLocale(savedLocale)
    }, [])

    const toggleLocale = () => {
        const newLocale = locale === 'zh' ? 'en' : 'zh'
        setLocale(newLocale)
        Cookies.set('locale', newLocale)
    }

    return (
        <LocaleContext.Provider value={{ locale, toggleLocale }}>
            {children}
        </LocaleContext.Provider>
    )
}

export function useLocale() {
    const context = useContext(LocaleContext)
    if (context === undefined) {
        throw new Error('useLocale must be used within a LocaleProvider')
    }
    return context
} 