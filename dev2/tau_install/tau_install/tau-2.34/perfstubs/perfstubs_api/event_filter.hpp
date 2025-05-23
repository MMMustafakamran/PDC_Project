/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <regex>
#include <vector>

namespace perfstubs {
namespace python {

class event_filter {
public:
    static bool exclude(const std::string &name, const std::string &filename);
    static event_filter& instance(void);
    bool have_filter;
    std::vector<std::regex> exclude_names;
    std::vector<std::regex> exclude_files;
    std::vector<std::regex> include_names;
    bool have_include_names;
    std::vector<std::regex> include_files;
    bool have_include_files;
private:
    /* Declare the constructor, only used by the "instance" method.
     * it is defined in the cpp file. */
    event_filter(void);
    ~event_filter(void) {};
    /* Disable the copy and assign methods. */
    event_filter(event_filter const&)    = delete;
    void operator=(event_filter const&)  = delete;
    bool _exclude_name(const std::string &name);
    bool _exclude_file(const std::string &filename);
    static event_filter * _instance;
    rapidjson::Document configuration;
};

}
}

