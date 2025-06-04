import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.InputMismatchException;
import java.io.IOException;
import java.io.InputStream;


public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        CMenorah solver = new CMenorah();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class CMenorah {
        public void solve(int testNumber, FastReader in, PrintWriter out) {
            int n = in.nextInt();
            char[] a = in.next().toCharArray();
            char[] b = in.next().toCharArray();

            int ans = checkPairs(a, b, -1);
            ans = Math.min(ans, changePair(a, b, '1'));
            ans = Math.min(ans, changePair(a, b, '0'));

            if (ans == Integer.MAX_VALUE) ans = -1;
            out.println(ans);
        }

        int changePair(char[] a, char[] b, char t) {
            int index = -1;
            int n = a.length;

            for (int i = 0; i < n; ++i) {
                if (a[i] == '1' && b[i] == t) {
                    index = i;
                    break;
                }
            }

            return checkPairs(a, b, index);
        }

        int checkPairs(char[] a, char[] b, int changeStringIndex) {
            int n = a.length;
            int val = 0;

            char[] tmp = new char[n];
            System.arraycopy(a, 0, tmp, 0, n);

            if (changeStringIndex != -1) {
                val = 1;

                for (int i = 0; i < n; ++i) {
                    if (i == changeStringIndex) continue;
                    tmp[i] = a[i] == '0' ? '1' : '0';
                }
            }

            int _10 = 0, _01 = 0;
            for (int i = 0; i < n; ++i) {
                if (tmp[i] != b[i]) {
                    if (tmp[i] == '0') _01++;
                    else _10++;
                }
            }

            return _01 == _10 ? 2 * _01 + val : Integer.MAX_VALUE;
        }

    }

    
}
