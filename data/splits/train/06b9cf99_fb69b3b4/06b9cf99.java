import java.util.*;
import java.io.*;

public class Solution {

    public static void main(String[] args) throws IOException {

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        int t = Integer.parseInt(br.readLine());

        BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));

        StringTokenizer st;

        while (t-- > 0) {

            // int n = Integer.parseInt(br.readLine());

            st = new StringTokenizer(br.readLine());
            int n = Integer.parseInt(st.nextToken());
            long k = Long.parseLong(st.nextToken());

            st = new StringTokenizer(br.readLine());

            long arr[] = new long[n];
            long tot_sum = 0;
            for(int i = 0; i < n; i++)
            {
            long e = Long.parseLong(st.nextToken());
            arr[i] = e;
            tot_sum+=e;
            }

            if(tot_sum <= k)
            {
                output.write("0\n");
                continue;
            }
            Arrays.sort(arr);
            long psum = 0;
            long res = tot_sum - k;
            for(int p = 0; p < n - 1; p++)
            {
                int ind = n - p - 1;
                psum+=arr[ind];
                long sum = arr[0] + psum - tot_sum + k;
                double d = sum / (double)(p + 2);
                long x = (long)Math.floor(d);
                // output.write("x = " + x + " sum = " + sum + " d = " + d + "\n");
                long numsteps = Math.max(arr[0] - x, 0) + p + 1;
                if(numsteps < 0)
                    continue;
                // output.write("for p = " + p + " the number of steps req = " + numsteps + " and the x = " + x + "\n");
                res = Math.min(res, numsteps);
            }

            output.write(res + "\n");

            // int k = Integer.parseInt(st.nextToken());

            // char arr[] = br.readLine().toCharArray();

            // output.write();
            // int n = Integer.parseInt(st.nextToken());

            // HashMap<Character, Integer> map = new HashMap<Character, Integer>();

            // if
            // output.write("YES\n");
            // else
            // output.write("NO\n");

            // long a = Long.parseLong(st.nextToken());
            // long b = Long.parseLong(st.nextToken());

            // if(flag == 1)
            // output.write("NO\n");
            // else
            // output.write("YES\n" + x + " " + y + " " + z + "\n");

            // output.write(n+ "\n");

        }

        output.flush();

    }

}
