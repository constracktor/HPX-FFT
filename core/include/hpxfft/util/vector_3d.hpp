#ifndef vector_3d_H_INCLUDED
#define vector_3d_H_INCLUDED

#include <hpx/serialization.hpp>

namespace hpxfft::util
{

template <typename T>
struct vector_3d
{
    T *values_;
    std::size_t size_;
    // row major format
    std::size_t n_x_;  // First dimension
    std::size_t n_y_;  // Second dimension
    std::size_t n_z_;  // Third dimension

  public:
    using iterator = T *;
    using const_iterator = const T *;
    // default constructor
    vector_3d();
    vector_3d(std::size_t n_x, std::size_t n_y, std::size_t n_z);
    // explicit contructors
    vector_3d(std::size_t n_x, std::size_t n_y, std::size_t n_z, const T &v);
    // copy constructor
    vector_3d(const vector_3d<T> &);
    // move constructor
    vector_3d(vector_3d<T> &&) noexcept;
    // destructor
    ~vector_3d();
    // operators
    vector_3d<T> &operator=(const vector_3d<T> &);
    vector_3d<T> &operator=(vector_3d<T> &&) noexcept;
    T &operator()(std::size_t i, std::size_t j, std::size_t k);
    const T &operator()(std::size_t i, std::size_t j, std::size_t k) const;
    T &at(std::size_t i, std::size_t j, std::size_t k);
    const T &at(std::size_t i, std::size_t j, std::size_t k) const;
    constexpr T *data() noexcept;
    constexpr const T *data() const noexcept;
    // iterators
    iterator begin() noexcept;
    const_iterator begin() const noexcept;
    iterator end() noexcept;
    const_iterator end() const noexcept;
    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept;
    iterator slice_yz(std::size_t k) noexcept;
    const_iterator slice_yz(std::size_t k) const noexcept;
    iterator vector_z(std::size_t i, std::size_t j) noexcept;
    const_iterator vector_z(std::size_t i, std::size_t j) const noexcept;
    // size
    std::size_t size() const noexcept;
    std::size_t n_x() const noexcept;
    std::size_t n_y() const noexcept;
    std::size_t n_z() const noexcept;
    void rearrange(std::size_t new_n_x, std::size_t new_n_y, std::size_t new_n_z);
    // Non-Member Functions
    template <typename H>
    friend bool operator==(const vector_3d<H> &lhs, const vector_3d<H> &rhs);

    // see https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    friend void swap(vector_3d &first, vector_3d &second)
    {
        std::swap(first.n_x_, second.n_x_);
        std::swap(first.n_y_, second.n_y_);
        std::swap(first.n_z_, second.n_z_);
        std::swap(first.size_, second.size_);
        std::swap(first.values_, second.values_);
    }

  private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void save(Archive &ar, const unsigned int) const
    {
        ar << n_x_ << n_y_ << n_z_ << size_;
        for (std::size_t i = 0; i < size_; ++i)
        {
            ar << values_[i];
        }
    }

    template <typename Archive>
    void load(Archive &ar, const unsigned int)
    {
        ar >> n_x_ >> n_y_ >> n_z_ >> size_;
        delete[] values_;  // free any prior buffer (nullptr-safe)
        values_ = nullptr;
        if (size_ > 0)
        {
            values_ = new T[size_];
        }
        for (std::size_t i = 0; i < size_; ++i)
        {
            ar >> values_[i];
        }
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <typename T>
inline vector_3d<T>::vector_3d()
{
    n_x_ = 0;
    n_y_ = 0;
    n_z_ = 0;
    size_ = 0;
    values_ = nullptr;
}

template <typename T>
inline vector_3d<T>::vector_3d(std::size_t n_x, std::size_t n_y, std::size_t n_z)
{
    n_x_ = n_x;
    n_y_ = n_y;
    n_z_ = n_z;
    size_ = n_x_ * n_y_ * n_z_;
    values_ = new T[size_];

    for (std::size_t i = 0; i < size_; ++i)
    {
        values_[i] = T();
    }
}

template <typename T>
inline vector_3d<T>::vector_3d(std::size_t n_x, std::size_t n_y, std::size_t n_z, const T &v)
{
    n_x_ = n_x;
    n_y_ = n_y;
    n_z_ = n_z;
    size_ = n_x_ * n_y_ * n_z_;
    values_ = new T[size_];
    std::fill(begin(), end(), v);
}

template <typename T>
inline vector_3d<T>::vector_3d(const vector_3d<T> &src) :
    values_(nullptr),
    size_(src.size_),
    n_x_(src.n_x_),
    n_y_(src.n_y_),
    n_z_(src.n_z_)
{
    values_ = (size_ > 0) ? new T[size_] : nullptr;
    std::copy(src.begin(), src.end(), begin());
}

template <typename T>
inline vector_3d<T>::vector_3d(vector_3d<T> &&mv) noexcept :
    values_(mv.values_),
    size_(mv.size_),
    n_x_(mv.n_x_),
    n_y_(mv.n_y_),
    n_z_(mv.n_z_)
{
    mv.values_ = nullptr;
    mv.size_ = 0;
    mv.n_x_ = 0;
    mv.n_y_ = 0;
    mv.n_z_ = 0;
}

template <typename T>
inline vector_3d<T>::~vector_3d()
{
    delete[] values_;
}

template <typename T>
inline vector_3d<T> &vector_3d<T>::operator=(const vector_3d<T> &src)
{
    if (this != &src)
    {
        vector_3d<T> tmp(src);
        swap(*this, tmp);
    }
    return *this;
}

template <typename T>
inline vector_3d<T> &vector_3d<T>::operator=(vector_3d<T> &&mv) noexcept
{
    if (this != &mv)
    {
        delete[] values_;
        values_ = mv.values_;
        size_ = mv.size_;
        n_x_ = mv.n_x_;
        n_y_ = mv.n_y_;
        n_z_ = mv.n_z_;

        mv.values_ = nullptr;
        mv.size_ = 0;
        mv.n_x_ = 0;
        mv.n_y_ = 0;
        mv.n_z_ = 0;
    }
    return *this;
}

template <typename T>
inline T &vector_3d<T>::operator()(std::size_t i, std::size_t j, std::size_t k)
{
    return values_[i * n_y_ * n_z_ + j * n_z_ + k];
}

template <typename T>
inline const T &vector_3d<T>::operator()(std::size_t i, std::size_t j, std::size_t k) const
{
    return values_[i * n_y_ * n_z_ + j * n_z_ + k];
}

template <typename T>
inline T &vector_3d<T>::at(std::size_t i, std::size_t j, std::size_t k)
{
    if ((i * n_y_ * n_z_ + j * n_z_ + k >= size_) || (i >= n_x_) || (j >= n_y_) || (k >= n_z_))
    {
        throw std::runtime_error("out of range exception");
    }
    else
    {
        return values_[i * n_y_ * n_z_ + j * n_z_ + k];
    }
}

template <typename T>
inline const T &vector_3d<T>::at(std::size_t i, std::size_t j, std::size_t k) const
{
    if ((i * n_y_ * n_z_ + j * n_z_ + k >= size_) || (i >= n_x_) || (j >= n_y_) || (k >= n_z_))
    {
        throw std::runtime_error("out of range exception");
    }
    else
    {
        return values_[i * n_y_ * n_z_ + j * n_z_ + k];
    }
}

template <typename T>
inline constexpr T *vector_3d<T>::data() noexcept
{
    return values_;
}

template <typename T>
inline constexpr const T *vector_3d<T>::data() const noexcept
{
    return values_;
}

template <typename T>
inline typename vector_3d<T>::iterator vector_3d<T>::begin() noexcept
{
    return values_;
}

template <typename T>
inline typename vector_3d<T>::const_iterator vector_3d<T>::begin() const noexcept
{
    return values_;
}

template <typename T>
inline typename vector_3d<T>::iterator vector_3d<T>::end() noexcept
{
    return values_ + size_;
}

template <typename T>
inline typename vector_3d<T>::const_iterator vector_3d<T>::end() const noexcept
{
    return values_ + size_;
}

template <typename T>
inline typename vector_3d<T>::const_iterator vector_3d<T>::cbegin() const noexcept
{
    return values_;
}

template <typename T>
inline typename vector_3d<T>::const_iterator vector_3d<T>::cend() const noexcept
{
    return values_ + size_;
}

template <typename T>
inline typename vector_3d<T>::iterator vector_3d<T>::slice_yz(std::size_t i) noexcept
{
    return values_ + i * n_y_ * n_z_;
}

template <typename T>
inline typename vector_3d<T>::const_iterator vector_3d<T>::slice_yz(std::size_t i) const noexcept
{
    return values_ + i * n_y_ * n_z_;
}

template <typename T>
inline typename vector_3d<T>::iterator vector_3d<T>::vector_z(std::size_t i, std::size_t j) noexcept
{
    return values_ + i * n_y_ * n_z_ + j * n_z_;
}

template <typename T>
inline typename vector_3d<T>::const_iterator vector_3d<T>::vector_z(std::size_t i, std::size_t j) const noexcept
{
    return values_ + i * n_y_ * n_z_ + j * n_z_;
}

template <typename T>
inline std::size_t vector_3d<T>::size() const noexcept
{
    return size_;
}

template <typename T>
inline std::size_t vector_3d<T>::n_x() const noexcept
{
    return n_x_;
}

template <typename T>
inline std::size_t vector_3d<T>::n_y() const noexcept
{
    return n_y_;
}

template <typename T>
inline std::size_t vector_3d<T>::n_z() const noexcept
{
    return n_z_;
}

template <typename T>
inline void vector_3d<T>::rearrange(std::size_t new_n_x, std::size_t new_n_y, std::size_t new_n_z)
{
    if (new_n_x * new_n_y * new_n_z != size_)
    {
        throw std::runtime_error("New dimensions do not match total size.");
    }
    n_x_ = new_n_x;
    n_y_ = new_n_y;
    n_z_ = new_n_z;
}

template <typename H>
inline bool operator==(const vector_3d<H> &lhs, const vector_3d<H> &rhs)
{
    if (lhs.n_x_ != rhs.n_x_ || lhs.n_y_ != rhs.n_y_ || lhs.n_z_ != rhs.n_z_)
    {
        return false;
    }

    for (std::size_t i = 0; i < lhs.size_; ++i)
    {
        if (lhs.values_[i] != rhs.values_[i])
        {
            return false;
        }
    }

    return true;
}

}  // namespace hpxfft::util
#endif  // vector_3d_H_INCLUDED
