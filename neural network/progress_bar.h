#pragma once

#include <iostream>
namespace __Progress_bar {
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#endif // _WIN32

	inline void _Show_console_cursor(const bool show) {
#if defined(_WIN32)
		static const HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_CURSOR_INFO cci;
		GetConsoleCursorInfo(handle, &cci);
		cci.bVisible = show; // show/hide cursor
		SetConsoleCursorInfo(handle, &cci);
#elif defined(__linux__)
		cout << (show ? "\033[?25h" : "\033[?25l"); // show/hide cursor
#endif // Windows/Linux
	}

	using std::cout;
	using std::endl;

	struct _Loading_bar_iterator;
	struct _Sentinel { };

	class progress_bar
	{
	public:
		using iterator = _Loading_bar_iterator;
		friend struct iterator;

		_CONSTEXPR17 progress_bar(size_t _MaxVal)
			: _Max(_MaxVal) { }

		inline void update(double val) {
			if (_Curr == _Max) {
				cout << endl;
				return;
			}
			_Show_console_cursor(false);

			int cp = GetConsoleOutputCP();
			SetConsoleOutputCP(65001);
			_Clear_line();
			++_Curr;
			_Update(val);

			SetConsoleOutputCP(cp);
			_Show_console_cursor(true);
		}
		constexpr size_t size() const noexcept {
			return _Max;
		}
		constexpr operator bool() const {
			return _Curr != _Max;
		}
		constexpr friend bool operator<(const size_t& _Val, const progress_bar& _Bar) {
			return _Val < _Bar._Max;
		}
		_CONSTEXPR17 auto begin();
		constexpr auto end() const {
			return _Sentinel{};
		}
		static constexpr char bg_char = ' ';
	private:
		inline void _Update(double x) const {
			cout << x << " [";
			int curr_percentage = _Curr * 100 / _Max;
			int bars_to_draw = curr_percentage * _Bar_length / 100;
			int spaces_left = _Bar_length - bars_to_draw;

			for (int i = 0; i < bars_to_draw; ++i)
				cout << _Bar_char;

			for (int i = 0; i < spaces_left; ++i)
				cout << bg_char;

			cout << "] " << _Curr << "/" << _Max;
		}
		inline void _Clear_line() const {
			cout << "\r";
			for (size_t i = 0; i < _Bar_length + 16; ++i)
				cout << bg_char;
			cout << "\r";
		}


		size_t _Curr = 0;
		const size_t _Max;
		static constexpr size_t _Bar_length = 40;
		static constexpr char _Bar_char[4] = { -30, -106, -120, 0 };
	};

	struct _Loading_bar_iterator {
		_CONSTEXPR17 _Loading_bar_iterator(progress_bar& _Bar)
			: _Bar(_Bar) { }

		inline size_t operator*() const {
			return _Bar._Curr;
		}
		inline _Loading_bar_iterator& operator++() {
			_Bar.update(0);
			return *this;
		}
		inline _Loading_bar_iterator operator++(int) {
			auto _Tmp = *this;
			_Bar.update(0);
			return _Tmp;
		}
		inline bool operator==(_Sentinel) const {
			return _Bar._Curr == _Bar._Max;
		}
		inline bool operator!=(_Sentinel) const {
			return _Bar._Curr != _Bar._Max;
		}

		progress_bar& _Bar;
	};

	inline _CONSTEXPR17 auto progress_bar::begin()
	{
		return iterator(*this);
	}
}

using __Progress_bar::progress_bar;