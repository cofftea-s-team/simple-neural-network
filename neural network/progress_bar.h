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
	inline void _Set_cursos_pos(int x, int y) {
#if defined(_WIN32)
		static const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
		COORD pos = { x, y };
		SetConsoleCursorPosition(output, pos);
#elif defined(__linux__)
		printf("\033[%d;%dH", y + 1, x + 1);
#endif // Windows/Linux
	}
	inline auto _Get_cursor_pos() {
#if defined(_WIN32)
		static const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		GetConsoleScreenBufferInfo(output, &csbi);
		return csbi.dwCursorPosition;
#elif defined(__linux__)
		printf("\033[6n");
		int rows, cols;
		scanf("\033[%d;%dR", &rows, &cols);
		return { cols - 1, rows - 1 };
#endif // Windows/Linux 
	}
#include <cstdlib>
#include <ctime>
	struct _Loading_bar_iterator;
	struct _Sentinel { };

	class progress_bar
	{
	public:
		using iterator = _Loading_bar_iterator;
		friend struct iterator;

		inline progress_bar(size_t _MaxVal)
			: _Max(_MaxVal) , _Begin_pos(_Get_cursor_pos())
		{
			_Show_console_cursor(false);
			_Prev_cp = GetConsoleOutputCP();
			SetConsoleOutputCP(65001);
			cout << '[';
			_Set_cursos_pos(_Bar_length + 1, _Begin_pos.Y);
			cout << ']';
			_Begin_time = clock();
		}

		inline bool update() {
			return update([]() -> void {});
		}

		template <class _Lambda, class... _FnArgs>
		inline bool update(_Lambda&& _Post_text_fn, _FnArgs&&... _Args) {
			_Clear_post_text();
			_Update();
			++_Curr;
			
			_Set_cursos_pos(_Bar_length + 3, _Begin_pos.Y);
			_Post_text_fn(forward<_FnArgs>(_Args)...);
			_Post_text_length = _Get_cursor_pos().X - _Begin_pos.X - _Bar_length - 1;
			
			if (_Curr == _Max) {
				_Finish();
				return false;
			}
			_Print_estimated_time();
			return true;
		}
		inline void clear() {
			_Set_cursos_pos(_Begin_pos.X, _Begin_pos.Y);
			for (int i = 0; i < _Bar_length + _Post_text_length + 4; ++i)
				cout << bg_char;
			_Clear_estimated_time();
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
		inline void _Update() const {
			int _Step = _Bar_length / _Max + 1;
			int _Curr_pos = _Curr * 100 / _Max * _Bar_length / 100 + 1;
			_Set_cursos_pos(_Curr_pos, _Begin_pos.Y);
			for (int i = 0; i < _Step; ++i) {
				cout << _Bar_char;
			}
			
		}
		inline void _Clear_post_text() const {
			_Set_cursos_pos(_Bar_length + 3, _Begin_pos.Y);
			for (size_t i = 0; i < _Post_text_length; ++i)
				cout << bg_char;
		}
		inline void _Print_estimated_time() const {
			_Set_cursos_pos(_Begin_pos.X, _Begin_pos.Y + 1);
			cout << "Estimated time: " << _Get_estimated_time() << " ms";
		}
		inline size_t _Get_estimated_time() const {
			clock_t _Curr_time = clock();
			return (_Curr_time - _Begin_time) * (_Max - _Curr + 1) / (_Curr + 1);
		}
		inline void _Clear_estimated_time() const {
			_Set_cursos_pos(_Begin_pos.X, _Begin_pos.Y + 1);
			for (size_t i = 0; i < 35; ++i)
				cout << bg_char;
		}
		inline void _Finish() const {
			_Clear_estimated_time();
			_Set_cursos_pos(_Begin_pos.X, _Begin_pos.Y);
			cout << endl;
			SetConsoleOutputCP(_Prev_cp);
			_Show_console_cursor(true);
		}
		int _Prev_cp;
		size_t _Curr = 0;
		const size_t _Max;
		static constexpr size_t _Bar_length = 50;
		static constexpr char _Bar_char[4] = { -30, -106, -120, 0 };
		size_t _Post_text_length = 0;
		const COORD _Begin_pos;
		clock_t _Begin_time;
	};

	struct _Loading_bar_iterator {
		_CONSTEXPR17 _Loading_bar_iterator(progress_bar& _Bar)
			: _Bar(_Bar) { }

		inline size_t operator*() const {
			return _Bar._Curr;
		}
		inline _Loading_bar_iterator& operator++() {
			_Bar.update([]() {});
			return *this;
		}
		inline _Loading_bar_iterator operator++(int) {
			auto _Tmp = *this;
			_Bar.update([]() {});
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