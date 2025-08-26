#ifndef vector_2d_H_INCLUDED
#define vector_2d_H_INCLUDED

#include <cstddef>
#include <algorithm> 

//#include <iostream>
#include <hpx/serialization.hpp>

template<typename T>
struct vector_2d
{
    T* values_;
    std::size_t size_;
    // row major format
    std::size_t n_row_; // First dimension
    std::size_t n_col_; // Second dimension 

public:
    using iterator = T*;
    using const_iterator = const T*;
    // default constructor
    vector_2d();
    vector_2d(std::size_t n_row, std::size_t n_col);
    // explicit contructors
    vector_2d(std::size_t n_row, std::size_t n_col, const T& v );
    // copy constructor
    vector_2d(const vector_2d<T>&);
    // move constructor
    vector_2d(vector_2d<T>&&) noexcept;
    // destructor
    ~vector_2d()=default;
    // operators
    vector_2d<T>& operator=(vector_2d<T>&);
    vector_2d<T>& operator=(vector_2d<T>&&) noexcept;
    T& operator()(std::size_t i, std::size_t j);
    const T& operator()(std::size_t i, std::size_t j) const;
    T& at(std::size_t i, std::size_t j);
    const T& at(std::size_t i, std::size_t j) const;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;
    // iterators
    iterator begin() noexcept;
    const_iterator begin() const noexcept;
    iterator end() noexcept;
    const_iterator end() const noexcept;
    const_iterator cbegin() const noexcept;
    const_iterator  cend() const noexcept;
    iterator row(std::size_t i) noexcept;
    const_iterator row(std::size_t i) const noexcept;
    // size
    std::size_t size() const noexcept;
    std::size_t n_row() const noexcept;
    std::size_t n_col() const noexcept;
    // Non-Member Functions
    template<typename H> friend bool operator==(const vector_2d<H>& lhs, const vector_2d<H>& rhs);
    // see https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    friend void swap(vector_2d& first, vector_2d& second)
    {
        std::swap(first.n_row_, second.n_row_);
        std::swap(first.n_col_, second.n_col_);
        std::swap(first.size_, second.size_);
        std::swap(first.values_, second.values_);
    }

private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar &n_row_;
        ar &n_col_;
        ar &size_;
        for(std::size_t i=0; i<size_; ++i)
        {
            ar& values_[i];
        }
        // clang-format on
    }
}; 

template<typename T>
inline vector_2d<T>::vector_2d()
{
    n_row_ = 0;
    n_col_ = 0;
    size_ = 0;
    values_ = nullptr;
}

template<typename T>
inline vector_2d<T>::vector_2d(std::size_t n_row, std::size_t n_col)
{
    n_row_ = n_row;
    n_col_ = n_col;
    size_ = n_row_ * n_col_;

    values_ = new T[size_];

    for(std::size_t i = 0; i < size_; ++i)
        values_[ i ] = T();
}

template<typename T>
inline vector_2d<T>::vector_2d(std::size_t n_row, std::size_t n_col, const T& v)
{
    n_row_ = n_row;
    n_col_ = n_col;
    size_ = n_row_ * n_col_;

    values_ = new T[size_];

    // for(std::size_t i = 0; i < size_; ++i)
    //     values_[ i ] = v;
    std::fill(begin(),end(),v);
}

template<typename T>
inline vector_2d<T>::vector_2d(const vector_2d<T>& src) : 
    n_row_(src.n_row_),
    n_col_(src.n_col_),
    size_(src.size_),
    values_(new T[size_])
{
    // for(std::size_t i = 0; i < size_; ++i)
    //     values_[ i ] = src.values_[ i ];
    // note that this is non-throwing
    std::copy(src.begin(), src.end(), begin());
}

template<typename T>
inline vector_2d<T>::vector_2d(vector_2d<T>&& mv) noexcept
{
    swap(*this, mv);
}

template<typename T>
inline vector_2d<T>& vector_2d<T>::operator=(vector_2d<T>& src)
{
    swap(*this, src);

    return *this;
}

template<typename T>
inline vector_2d<T>& vector_2d<T>::operator=(vector_2d<T>&& mv) noexcept
{
    swap(*this, mv);

    return *this;
}

template<typename T>
inline typename vector_2d<T>::iterator vector_2d<T>::begin() noexcept
{
    return values_;
}

template<typename T>
inline typename vector_2d<T>::const_iterator vector_2d<T>::begin() const noexcept
{
    return values_;
}

template<typename T>
inline typename vector_2d<T>::iterator  vector_2d<T>::end() noexcept
{
    return values_ + size_;
}

template<typename T>
inline typename vector_2d<T>::const_iterator  vector_2d<T>::end() const noexcept
{
    return values_ + size_;
}

template<typename T>
inline typename vector_2d<T>::const_iterator vector_2d<T>::cbegin() const noexcept
{
    return values_;
}

template<typename T>
inline typename vector_2d<T>::const_iterator  vector_2d<T>::cend() const noexcept
{
    return values_ + size_;
}

template<typename T>
inline typename vector_2d<T>::iterator vector_2d<T>::row(std::size_t i) noexcept
{
    return values_ + i * n_col_;
}

template<typename T>
inline typename vector_2d<T>::const_iterator vector_2d<T>::row(std::size_t i) const noexcept
{
    return values_ + i * n_col_;
}

template<typename T>
inline T& vector_2d<T>::operator( ) (std::size_t i, std::size_t j)
{
    return values_[ i * n_col_ + j ];
}

template<typename T>
inline T& vector_2d<T>::at (std::size_t i, std::size_t j)
{
    if(i * n_col_ + j  >= size_)
        throw std::runtime_error("out of range exception");
    else
        return values_[ i * n_col_ + j ];
}

template<typename T>
inline const T& vector_2d<T>::operator( ) (std::size_t i, std::size_t j) const
{
    return values_[ i * n_col_ + j ];
}

template<typename T>
inline const T& vector_2d<T>::at (std::size_t i, std::size_t j) const
{
    if(i * n_col_ + j >= size_)
        throw std::runtime_error("out of range exception");
    else
        return values_[ i * n_col_ + j ];
}

template<typename T>
inline constexpr T* vector_2d<T>::data() noexcept
{
    return values_;
}

template<typename T>
inline constexpr const T* vector_2d<T>::data() const noexcept
{
    return values_;
}

template<typename T>
inline std::size_t  vector_2d<T>::size() const noexcept
{
    return size_;
}

template<typename T>
inline std::size_t  vector_2d<T>::n_row() const noexcept
{
    return n_row_;
}

template<typename T>
inline std::size_t  vector_2d<T>::n_col() const noexcept
{
    return n_col_;
}

template<typename H>
inline bool operator==(const vector_2d<H>& lhs, const vector_2d<H>& rhs)
{
    if(lhs.n_row_ != rhs.n_row_ || lhs.n_col_ != rhs.n_col_)
        return false;

    for(std::size_t i = 0; i < lhs.size_; ++i)
        if(lhs.values_[ i ] != rhs.values_[ i ])
            return false;

    return true;
}
#endif // vector_2d_H_INCLUDED
