import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

public class Main {

    private static void run() throws IOException {
        int n = in.nextInt();
        int[] a = new int[n];

        int[] ans = new int[n];

        int sum = 0;
        a[0] = in.nextInt();
        for (int i = 1; i < n; i++) {
            a[i] = in.nextInt();

            if (a[i] * sum <= 0) {
                ans[i] = 1;
            } else {
                ans[i] = -1;
            }
            sum += ans[i] * a[i];
        }
        if (sum == 0) {
            sum -= ans[1] * a[1];
            ans[1] = -ans[1];
            sum += ans[1] * a[1];
        }

        int d = gcd(Math.abs(sum), Math.abs(a[0]));

        if (a[0] * sum < 0) {
            ans[0] = Math.abs(sum) / d;
        } else {
            ans[0] = -Math.abs(sum) / d;
        }
        for (int i = 1; i < n; i++) {
            ans[i] *= Math.abs(a[0]) / d;
        }

        print_array(ans);
    }

    public static void main(String[] args) throws IOException {
        in = new Reader();
        out = new PrintWriter(new OutputStreamWriter(System.out));

        int t = in.nextInt();
        for (int i = 0; i < t; i++) {
            run();
        }

        out.flush();
        in.close();
        out.close();
    }

    private static int gcd(int a, int b) {
        if (a == 0 || b == 0)
            return 0;
        while (b != 0) {
            int tmp;
            tmp = a % b;
            a = b;
            b = tmp;
        }
        return a;
    }


    

    

    private static Reader in;
    private static PrintWriter out;

    

    

    private static void print_array(int[] array) {
        for (int now : array) {
            out.print(now);
            out.print(' ');
        }
        out.println();
    }
    
}
