import java.util.*;
import java.lang.*;
import java.io.*;


public class Main {
    void solve() {
        int n = in.nextInt();
        char[] a = in.nextLine().toCharArray();
        char[] b = in.nextLine().toCharArray();

        int ans = MAX;
        ans = Math.min(ans, operations(a, b));
        ans = Math.min(ans, change(a, b, '1'));
        ans = Math.min(ans, change(a, b, '0'));

        if (ans == MAX)ans = -1;
        out.append(ans + "\n");
    }

    int operations(char[] a, char[] b) {
        int count01 = 0 , count10 = 0;
        int n = a.length;
        for (int i = 0 ; i < n; i++) {
            if (a[i] != b[i]) {
                if (a[i] == '0')count01++;
                else count10++;
            }
        }
        if (count01 != count10)return MAX;
        return count01 + count10;
    }

    int change(char[] a, char[] b , char ch) {
        int n = a.length;
        char[] c = new char[n];
        for (int i = 0 ; i < n; i++)c[i] = a[i];
        int index = -1;
        for (int i = 0; i < n; i++) {
            if (c[i] == '1' && b[i] == ch) {
                index = i;
                break;
            }
        }
        if (index == -1)return MAX;
        for (int i = 0 ; i < n; i++) {
            if (i == index)continue;
            c[i] = (char)( '0' + ('1' - c[i]) );
        }
        int  local = operations(c, b);
        if (local == MAX)return MAX;
        return 1 + local;
    }

    public static void main (String[] args) {
        // It happens - Syed Mizbahuddin
        Main sol = new Main();
        int t = 1;
        t = in.nextInt();
        while (t-- != 0) {
            sol.solve();
        }
        System.out.print(out);
    }

    

    

    

    

    

    

    

    

    

    static {
        try {
            System.setIn(new FileInputStream("input.txt"));
            System.setOut(new PrintStream(new FileOutputStream("output.txt")));
            err = new PrintStream(new FileOutputStream("error.txt"));
        } catch (Exception e) {}
    }

    static FastReader in;
    static StringBuilder out;
    static PrintStream err;
    final int MAX;
    final int MIN;
    int mod ;
    Main() {
        in = new FastReader();
        out = new StringBuilder();
        MAX = Integer.MAX_VALUE;
        MIN = Integer.MIN_VALUE;
        mod = (int)1e9 + 7;
    }
    

    
}
