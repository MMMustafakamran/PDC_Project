/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <string>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <execinfo.h>
#include <unistd.h>
#include <sys/ucontext.h>
#include "thread_instance.hpp"
#include "address_resolution.hpp"
#include <errno.h>
#include <string.h>
#include <regex>
#include "utils.hpp"

void apex_print_backtrace() {
  void *trace[32];
  size_t size, i;
  char **strings;

  size    = backtrace( trace, 32 );
  /* overwrite sigaction with caller's address */
  //trace[1] = (void *)ctx.eip;
  strings = backtrace_symbols( trace, size );

  std::cerr << std::endl;
  std::cerr << "BACKTRACE (backtrace_symbols):";
  std::cerr << std::endl;
  std::cerr << std::endl;

  char exe[256];
  int len = readlink("/proc/self/exe", exe, 256);
  if (len != -1) {
    exe[len] = '\0';
  }

  // skip the first frame, it is this handler
  for( i = 1; i < size; i++ ){
   std::cerr << strings[i] << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "BACKTRACE (binutils):";
  std::cerr << std::endl;
  std::cerr << std::endl;

  // skip the first frame, it is this handler
  for( i = 1; i < size; i++ ){
#if defined(APEX_HAVE_BFD) || defined(__APPLE__)
   std::cerr << *(apex::lookup_address((uintptr_t)trace[i], true, true)) << std::endl;
#else
   char syscom[1024] = {0};
   sprintf(syscom,"addr2line -f -i -e %s %p", exe, trace[i]);
   int retval = system(syscom);
   // do nothing, we're exiting anyway
   if (retval != 0) { continue; }
#endif
  }
}

static void apex_custom_signal_handler(int sig) {

  int errnum = errno;

  //static std::mutex output_mutex;
  //std::unique_lock<std::mutex> l(output_mutex);
  fflush(stderr);
  std::cerr << std::endl;
  apex_print_backtrace();
  std::cerr << std::endl;

  if (apex::apex::instance() != nullptr) {
    std::cerr << "********* Node " << apex::apex::instance()->get_node_id() <<
                ", Thread " << apex::thread_instance::get_id() << " " <<
                strsignal(sig) << " *********" << std::endl;
    std::cerr << std::endl;
    if(errnum) {
        std::cerr << "Value of errno: " << errno << std::endl;
        perror("Error printed by perror");
        std::cerr << "Error string: " << strerror( errnum ) << std::endl;
    }
  }

  std::cerr << "***************************************";
  std::cerr << std::endl;
  std::cerr << std::endl;
  fflush(stderr);
  //apex::finalize();
    _exit(sig);
}

#ifdef APEX_BE_GOOD_CITIZEN
std::map<int, struct sigaction> other_handlers;
#endif

static void apex_custom_signal_handler_advanced(int sig, siginfo_t * info, void * context) {
    apex_custom_signal_handler(sig);
#ifdef APEX_BE_GOOD_CITIZEN
    // call the old handler
    other_handlers[sig].sa_sigaction(sig, info, context);
#else
    APEX_UNUSED(info);
    APEX_UNUSED(context);
#endif
}

int apex_register_signal_handler() {
    static bool doOnce{false};
    if (doOnce) return 0;
  if (apex::test_for_MPI_comm_rank(0) == 0) {
    std::cout << "APEX signal handler registering..." << std::endl;
  }
  struct sigaction act;
  struct sigaction old;
  memset(&act, 0, sizeof(act));
  memset(&old, 0, sizeof(old));

  sigemptyset(&act.sa_mask);
  std::array<int,13> mysignals = {
    SIGHUP,
    SIGINT,
    SIGQUIT,
    SIGILL,
    //SIGTRAP,
    SIGIOT,
    SIGBUS,
    SIGFPE,
    SIGKILL,
    SIGSEGV,
    SIGABRT,
    SIGTERM,
    SIGXCPU,
    SIGXFSZ
  };
  //act.sa_handler = apex_custom_signal_handler;
  act.sa_flags = SA_RESTART | SA_SIGINFO;
  act.sa_sigaction = apex_custom_signal_handler_advanced;
  for (auto s : mysignals) {
    sigaction(s, &act, &old);
#ifdef APEX_BE_GOOD_CITIZEN
    other_handlers[s] = old;
#endif
  }
  if (apex::test_for_MPI_comm_rank(0) == 0) {
    std::cout << "APEX signal handler registered!" << std::endl;
  }
  doOnce = true;
  return 0;
}

void apex_test_signal_handler() {
  apex_custom_signal_handler(1);
}

std::vector<apex::profiler*>& profilers_to_exit(void) {
    static std::vector<apex::profiler*> _thevector;
    return _thevector;
}

std::atomic<size_t> threads_to_exit_count{0};

//static void apex_custom_signal_handler_thread_exit([[maybe_unused]] int sig) {
static void apex_custom_signal_handler_thread_exit(
    [[maybe_unused]] int sig,
    [[maybe_unused]] siginfo_t * info,
    [[maybe_unused]] void * context) {
    APEX_ASSERT(sig == SIGUSR2);
    auto p = apex::thread_instance::instance().get_current_profiler();
    apex::profiler* parent = nullptr;
    while(p != nullptr) {
        if (p->untied_parent == nullptr || p->untied_parent->tt_ptr->state != apex::task_wrapper::RUNNING) {
            parent = nullptr;
        } else {
            parent = p->untied_parent;
        }
        // only push profilers that were started on THIS thread...
        if (p != nullptr && p->thread_id == apex::thread_instance::instance().get_id()) {
            profilers_to_exit().push_back(p);
        }
        p = parent;
    }
    threads_to_exit_count--;
    return;
}

int apex_register_thread_cleanup(void) {
    static bool doOnce{false};
    if (doOnce) return 0;
    struct sigaction act;
    struct sigaction old;
    memset(&act, 0, sizeof(act));
    memset(&old, 0, sizeof(old));
    sigemptyset(&act.sa_mask);
    std::array<int,1> mysignals = { SIGUSR2 };
    act.sa_flags = SA_RESTART | SA_SIGINFO;
    act.sa_sigaction = apex_custom_signal_handler_thread_exit;
    for (auto s : mysignals) {
        sigaction(s, &act, &old);
#ifdef APEX_BE_GOOD_CITIZEN
        other_handlers[s] = old;
#endif
    }
    doOnce = true;
    return 0;
}

void apex_signal_all_threads(void) {
    auto tids = apex::thread_instance::gettids();
    pthread_t me = pthread_self();
    // generous...but not a hard limit. Trying not to allocate memory during the signal handler, above.
    profilers_to_exit().reserve(tids.size() * 10);
    // don't include myself
    threads_to_exit_count = tids.size() - 1;
    for (auto t : tids) {
        if (t != me) {
            pthread_kill(t,SIGUSR2);
        }
    }
    while(threads_to_exit_count > 0) { //wait!
    }
    for (auto p : profilers_to_exit()) {
        apex::stop(p);
    }
    profilers_to_exit().clear();
}
