import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Main
{
    public static void main(String[] args)
    {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        Scanner in = new Scanner(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskC solver = new TaskC();
        solver.solve(1, in, out);
        out.close();
    }

    static class TaskC
    {
        public void solve(int testNumber, Scanner in, PrintWriter out)
        {
            int T = in.nextInt();
            while (T-- > 0)
            {
                solveOne(in, out);
            }
        }

        private void solveOne(Scanner in, PrintWriter out)
        {
            int N = in.nextInt();
            int nums[] = L.readIntArray(N, in);
            int min[] = new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE};
            int rem[] = new int[]{N, N};
            long sum = 0;
            long ans = Long.MAX_VALUE;
            for (int idx = 0; idx < N; idx++)
            {
                min[idx % 2] = Math.min(min[idx % 2], nums[idx]);
                rem[idx % 2]--;
                sum += nums[idx];

                long cur = sum + rem[0] * (long) min[0] + rem[1] * (long) min[1];
                ans = Math.min(ans, cur);
            }
            out.println(ans);
        }

    }

    static class L
    {
        public static int[] readIntArray(int size, Scanner in)
        {
            int[] array = new int[size];
            for (int i = 0; i < size; i++)
            {
                array[i] = in.nextInt();
            }
            return array;
        }

    }
}
