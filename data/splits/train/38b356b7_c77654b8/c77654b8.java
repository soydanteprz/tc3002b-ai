import java.io.PrintWriter;
import java.util.*;
public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int tc = sc.nextInt();
        while(tc-->0){
            int n = sc.nextInt();
            int[] arr = new int[n]; for(int i = 0; i<n; i++)arr[i] = sc.nextInt();
            if(n % 2 == 0){
                for(int i = 0; i<n; i+=2){
                    pw.print((-arr[i + 1]) + " " + arr[i] + " ");
                }
                pw.println();
            }else{
                if(arr[0] + arr[1] != 0) pw.print(arr[2] + " " + arr[2] + " " + (-(arr[0] + arr[1])) + " ");
                else if(arr[0] + arr[2] != 0)pw.println(arr[1] + " " + (-(arr[0] + arr[2]))+" " + arr[1]);
                else pw.println(-(arr[1] + arr[2]) + " " + arr[0]+" " + arr[0]);
                for(int i = 3; i<n; i+=2){
                    pw.print((-arr[i + 1]) + " " + arr[i] + " ");
                }
                pw.println();
            }
        }
        pw.flush();
    }
}
