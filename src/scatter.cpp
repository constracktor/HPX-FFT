//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/iostream.hpp>


#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::collectives;

using vector = std::vector<std::uint32_t, std::allocator<std::uint32_t>>;
using split_vector = std::vector<vector>;

constexpr char const* scatter_direct_basename = "/test/scatter_direct/";

void test_multiple_use_with_generation()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_LTE(std::uint32_t(2), num_localities);//does not run on one locality
    std::uint32_t this_locality = hpx::get_locality_id();

    auto scatter_direct_client =
        hpx::collectives::create_communicator(scatter_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));


    // do stuff with vector
    std::uint32_t size=4;
    std::uint32_t sub_size=size/num_localities;
    std::uint32_t N = 1;

    std::vector<vector> values;
    std::vector<split_vector> values_div(N);

    // create values and do stuff
    for(std::uint32_t i = 0; i != N; ++i)
    {
        vector v(size);
        std::fill(v.begin(), v.end(), this_locality);
        //std::iota(v.begin(), v.end(), 0+10*this_locality);
        values.push_back(v);
    }

    // divide value vector
    for(std::uint32_t i = 0; i != N; ++i)
    {
        for (std::size_t j = 0; j != num_localities; ++j)
        {
            vector tmp(std::make_move_iterator(values[i].begin()+j*sub_size),
                                std::make_move_iterator(values[i].begin()+(j+1)*sub_size)); // move;
            values_div[i].push_back(tmp);
        }
    }




    std::vector<hpx::future<vector>> r2;
    for(std::uint32_t other_locality; other_locality != num_localities;++other_locality)
    {
        if(this_locality != other_locality)
        {
            
            for(std::uint32_t i = 0; i != N; ++i)
            {
                hpx::future<vector> result = scatter_from<vector>(
                    scatter_direct_client, generation_arg(other_locality+1));
                // extract from loop
                r2.push_back(std::move(result));
                //r2[i] = std::move(result);
            }
        }
    }


    std::vector<vector> r;//1   
    for(std::uint32_t i = 0; i != N; ++i)
    {
        hpx::future<vector> result = scatter_to(
            scatter_direct_client, std::move(values_div[i]), generation_arg(this_locality+1));
        r.push_back(result.get());
    }

    std::vector<vector> r3(num_localities);
    for(std::uint32_t other_locality; other_locality != num_localities;++other_locality)
    {
        if(this_locality != other_locality)
        {
            r3[other_locality] =r2[0].get(); 
        }
        else
        {
            r3[this_locality] =r[0];
        }
    }

    char const* msg = "\nLocality {1}:";
    hpx::util::format_to(hpx::cout, msg, this_locality)
             << std::flush;
    for(std::uint32_t i=0; i != num_localities;++i)
    {
            char const* msg = "\nFrom locality {1}: ";
            hpx::util::format_to(hpx::cout, msg, i)
             << std::flush;
            for (auto v : r3[i])
            {
                char const* msg = "{1} - ";
                hpx::util::format_to(hpx::cout, msg, v)
                    << std::flush;
            }
    }


        for (auto vec : r3)
        {
            for (auto v : vec)
            {
        
            char const* msg = "Locality {1} r: {2}\n";
            hpx::util::format_to(hpx::cout, msg, this_locality, v)
                << std::flush;
            }
        }
    








    // // communication
    // if (this_locality == 0)
    // {

    //     split_vector r;   
    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         hpx::future<vector> result = scatter_to(
    //             scatter_direct_client, std::move(values_div[i]), generation_arg(i+1));
    //         r.push_back(result.get());
    //     }

    //     std::vector<hpx::future<vector>> r2(N);
    //     split_vector r3; 
    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         hpx::future<vector> result = scatter_from<vector>(
    //             scatter_direct_client, generation_arg(i+1+N));
    //         // extract from loop
    //         r3.push_back(result.get());
    //        // r2[i] = std::move(result);
    //     }

    //     // for(std::uint32_t i = 0; i != N; ++i)
    //     // {
    //     //     r3.push_back(r2[i].get());
    //     // }









    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         for (auto v : r[i])
    //         {
    //             char const* msg = "Locality {1} r: {2}\n";
    //             hpx::util::format_to(hpx::cout, msg, this_locality, v)
    //                 << std::flush;
    //         }
    //     }
    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         for (auto v : r3[i])
    //         {
    //             char const* msg = "Locality {1} r: {2}\n";
    //             hpx::util::format_to(hpx::cout, msg, this_locality, v)
    //                 << std::flush;
    //         }
    //     }

    // }
    // else
    // {

    //     std::vector<hpx::future<vector>> r;
    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         hpx::future<vector> result = scatter_from<vector>(
    //             scatter_direct_client, generation_arg(i+1));
    //         // extract from loop
    //         r.push_back(std::move(result));
    //     }

    //     split_vector r2;   
    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         hpx::future<vector> result = scatter_to(
    //             scatter_direct_client, std::move(values_div[i]), generation_arg(i+1+N));
    //         r2.push_back(result.get());

    //     }








    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         for (auto v : r[i].get())
    //         {
    //             char const* msg = "Locality {1} r: {2}\n";
    //             hpx::util::format_to(hpx::cout, msg, this_locality, v)
    //                 << std::flush;
    //         }
    //     }
    //     for(std::uint32_t i = 0; i != N; ++i)
    //     {
    //         for (auto v : r2[i])
    //         {
    //             char const* msg = "Locality {1} r: {2}\n";
    //             hpx::util::format_to(hpx::cout, msg, this_locality, v)
    //                 << std::flush;
    //         }
    //     }

    // }












    // // test functionality based on immediate local result value
    // for (std::uint32_t i = 0; i != 10; ++i)
    // {
    //     if (this_locality == 0)
    //     {
    //         std::vector<std::uint32_t> data(num_localities);
    //         std::iota(data.begin(), data.end(), 42 + i);

    //         hpx::future<std::uint32_t> result = scatter_to(
    //             scatter_direct_client, std::move(data), generation_arg(i + 1));

    //         HPX_TEST_EQ(i + 42 + this_locality, result.get());
    //     }
    //     else
    //     {
    //         hpx::future<std::uint32_t> result = scatter_from<std::uint32_t>(
    //             scatter_direct_client, generation_arg(i + 1));

    //         HPX_TEST_EQ(i + 42 + this_locality, result.get());
    //     }
    // }
}

int hpx_main()
{
    // test_one_shot_use();
    // test_multiple_use();
    test_multiple_use_with_generation();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

#endif

// void test_one_shot_use()
// {
//     std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
//     HPX_TEST_LTE(std::uint32_t(2), num_localities);

//     std::uint32_t this_locality = hpx::get_locality_id();

//     // test functionality based on immediate local result value
//     for (std::uint32_t i = 0; i != 10; ++i)
//     {
//         if (this_locality == 0)
//         {
//             std::vector<std::uint32_t> data(num_localities);
//             std::iota(data.begin(), data.end(), 42 + i);

//             hpx::future<std::uint32_t> result =
//                 scatter_to(scatter_direct_basename, std::move(data),
//                     num_sites_arg(num_localities), this_site_arg(this_locality),
//                     generation_arg(i + 1));

//             HPX_TEST_EQ(i + 42 + this_locality, result.get());
//         }
//         else
//         {
//             hpx::future<std::uint32_t> result =
//                 scatter_from<std::uint32_t>(scatter_direct_basename,
//                     this_site_arg(this_locality), generation_arg(i + 1));

//             HPX_TEST_EQ(i + 42 + this_locality, result.get());
//         }
//     }
// }

// void test_multiple_use()
// {
//     std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
//     HPX_TEST_LTE(std::uint32_t(2), num_localities);

//     std::uint32_t this_locality = hpx::get_locality_id();

//     auto scatter_direct_client =
//         hpx::collectives::create_communicator(scatter_direct_basename,
//             num_sites_arg(num_localities), this_site_arg(this_locality));

//     // test functionality based on immediate local result value
//     for (std::uint32_t i = 0; i != 10; ++i)
//     {
//         if (this_locality == 0)
//         {
//             std::vector<std::uint32_t> data(num_localities);
//             std::iota(data.begin(), data.end(), 42 + i);

//             hpx::future<std::uint32_t> result =
//                 scatter_to(scatter_direct_client, std::move(data));

//             HPX_TEST_EQ(i + 42 + this_locality, result.get());
//         }
//         else
//         {
//             hpx::future<std::uint32_t> result =
//                 scatter_from<std::uint32_t>(scatter_direct_client);

//             HPX_TEST_EQ(i + 42 + this_locality, result.get());
//         }
//     }
// }