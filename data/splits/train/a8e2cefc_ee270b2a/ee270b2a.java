import java.util.*;

public class D{
    private static Scanner scanner = new Scanner(System.in);
    public static void main(String[] args){
        int q = scanner.nextInt();
        while(q-- > 0){
            int n = scanner.nextInt(),
            k = scanner.nextInt();
            int[] a = new int[k];
            for(int i=0;i<k;i++){
                a[i] = scanner.nextInt();
            }
            int[] t = new int[k];
            for(int j=0;j<k;j++){
                t[j] = scanner.nextInt();
            }
            long[] L = new long[n];
            long[] R = new long[n];
            for(int i=0;i<n;i++){
                L[i] = Integer.MAX_VALUE;
                R[i] = Integer.MAX_VALUE;
            }
            for(int i=0;i<k;i++){
                L[a[i]-1] = t[i];
                R[a[i]-1] = t[i];
            }
            
            long min = Integer.MAX_VALUE;
            for(int i=0;i<n;i++){
                L[i] = Math.min(min+1,L[i]);
                min = L[i];
            }
            min = Integer.MAX_VALUE;
            for(int i=n-1;i>=0;i--){
                R[i] = Math.min(min+1,R[i]);
                min = R[i];
            }
            for(int i=0;i<n;i++){
                System.out.print(Math.min(L[i],R[i]) + " ");
            }
            System.out.println();
        }
    }
}
