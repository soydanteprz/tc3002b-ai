import java.util.Scanner;

public class Subsequence {
	private static Scanner sc = new Scanner(System.in);
	
	public static void main(String args[]) {
		int t = sc.nextInt();
		
		while(t-->0) {
			int n = sc.nextInt();
			
			int a[] = new int[n];
			int b[] =  new int[n];
			
			for(int i=0;i<n;i++) {
				a[i]= sc.nextInt();
				
			}
			
			if(n%2==0) {
				calculateB(a,b,n);
			}
			else {
				calculateB(a,b,n-3);
				if (a[n - 2] + a[n - 3] != 0) {
					b[n - 3] = -a[n - 1];
					b[n - 2] = -a[n - 1];
					b[n - 1] = a[n - 2] + a[n - 3];
				} else if (a[n - 2] + a[n - 1] != 0) {
					b[n - 3] = a[n - 2] + a[n - 1];
					b[n - 2] = -a[n - 3];
					b[n - 1] = -a[n - 3];
				} else {
					b[n - 3] = -a[n - 2];
					b[n - 2] = a[n - 3] + a[n - 1];
					b[n - 1] = -a[n - 2];
				}
			}
			
			for(int i=0;i<n;i++) {
				System.out.print(b[i] + " ");
			}
			System.out.println();
		}

	}

	private static void calculateB(int[] a, int[] b, int n) {
		
		for(int i=0;i<n-1;i=i+2) {
			b[i] = -a[i+1];
			b[i+1] = a[i];
		}
		
	}
}
