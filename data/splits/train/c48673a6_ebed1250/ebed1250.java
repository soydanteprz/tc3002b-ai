import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

public class Main {

    private static void run() throws IOException {
        int n = in.nextInt();
        long[] c = new long[n];
        for (int i = 0; i < n; i++) {
            c[i] = in.nextInt();
        }

        long ans = (c[0] + c[1]) * n;
        long sum = c[0] + c[1];
        long[] min = {c[0], c[1]};
        for (int i = 2; i < n; i++) {
            sum += c[i];
            int index = i % 2;
            min[index] = Math.min(min[index], c[i]);

            int[] times = new int[2];
            times[index] = n - (i / 2 + 1);
            times[index ^ 1] = n - ((i - 1) / 2 + 1);
            ans = Math.min(ans, sum + min[0] * times[0] + min[1] * times[1]);
        }
        out.println(ans);
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

    

    static final long mod = 1000000007;

    

    

    @SuppressWarnings("FieldCanBeLocal")
    private static Reader in;
    private static PrintWriter out;

    

    

    
}
