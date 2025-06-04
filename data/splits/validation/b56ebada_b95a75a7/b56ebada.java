import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.*;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class First {

    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskA solver = new TaskA();
        //int a = 1;
        int t;
        t = in.nextInt();
        //t = 1;
        while (t > 0) {
            //out.print("Case #"+(a++)+": ");
            solver.call(in, out);
            t--;
        }
        out.close();

    }

    static class TaskA {
        public void call(InputReader in, PrintWriter out) {
            int n;
            n = in.nextInt();

            int[] arr = new int[n];
            Integer[] lower = new Integer[n];
            Integer[] higher = new Integer[n];

            for (int i = 0; i < n; i++) {
                arr[i] = in.nextInt();
            }

            TreeSet<Integer> set = new TreeSet<>();
            for (int i = 0; i < n; i++) {
                set.add(arr[i]);
                lower[i] = set.lower(arr[i]);
                higher[i] = set.higher(arr[i]);
            }

            for (int i = 1; i < n; i++) {
                if(arr[i]>arr[i-1]){
                    if(higher[i-1] != null && higher[i-1]<arr[i]){
                        out.println("NO");
                        return;
                    }
                }
                else{
                    if(lower[i-1] != null && lower[i-1]>arr[i]){
                        out.println("NO");
                        return;
                    }
                }
            }
            out.println("YES");


        }
    }

    

    

    

    

    

    

    static final Random random=new Random();

    

    
}
