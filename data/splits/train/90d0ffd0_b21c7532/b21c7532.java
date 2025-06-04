import java.util.*;
import java.math.*;
import java.io.*;

public class B{
	public static void main(String[] args){
		FastReader in = new FastReader(); 
		int t = in.nextInt();
		// int t = 1;
		while(t-- != 0){
			int n = in.nextInt(), m = in.nextInt();
		 	
		 	int[][] arr = new int[n][m];
		 	
		 	for(int i = 0; i < n; i++){
		 		String s = in.next();
		 		for(int j = 0; j < m; j++){
		 			arr[i][j] = s.charAt(j)-'0';
		 		}
		 	}

		 	// for(int i = 0; i < n; i++){
		 	// 	System.out.println(Arrays.toString(arr[i]));
		 	// }
		 	int res = 0;
		 	for(int i = 0; i < n-1; i += 2){
		 		for(int j = 0; j < m-1; j += 2){
		 			int r = arr[i][j] + arr[i][j+1] + arr[i+1][j] + arr[i+1][j+1];
		 			if(r == 1) res += 3;
		 			else if(r == 2) res += 2;
		 			else if(r == 3) res += 1;
		 			else if(r == 4) res += 4;
		 		}
		 	}
		 	
		 	if(n % 2 == 1){
		 		for(int j = 0; j < m-1; j += 2){
		 			int r = arr[n-1][j] + arr[n-1][j+1];
		 			if(r == 1) res += 3;
		 			else if(r == 2) res += 2;
		 		}
		 	}
		 	if(m % 2 == 1){
		 		for(int i = 0; i < n-1; i += 2){
		 			int r = arr[i][m-1] + arr[i+1][m-1];
		 			if(r == 1) res += 3;
		 			else if(r == 2) res += 2;
		 		}	
		 	}

		 	if(n % 2 == 1 && m % 2 == 1){
		 		if(arr[n-1][m-1] == 1) res+= 3;
		 	}

		 	System.out.println(res);

		 	for(int i = 0; i < n-1; i += 2){
		 		for(int j = 0; j < m-1; j += 2){
		 			int a = arr[i][j], b = arr[i][j+1], c = arr[i+1][j], d = arr[i+1][j+1];
		 			int r = a+b+c+d;
		 			if(r == 1){
		 				print1(i+1, j+1, a,b,c,d);
		 			}
		 			else if(r == 2){
		 				print2(i+1, j+1, a,b,c,d);
		 			}
		 			else if(r == 3){
		 				print3(i+1, j+1, a,b,c,d);
		 			}	
		 			else if(r == 4){
		 				print4(i+1, j+1, a,b,c,d);
		 			}
		 		}
		 	}

		 	// System.out.println("AFTER");
		 	if(n % 2 == 1){
		 		for(int j = 0; j < m-1; j += 2){
		 			int r = arr[n-1][j] + arr[n-1][j+1];
		 			int a = 0, b = 0, c = arr[n-1][j], d=arr[n-1][j+1];
		 			if(r == 1) print1(n-1,j+1, a,b,c,d);
		 			else if(r==2)print2(n-1,j+1, a,b,c,d);
		 			// System.out.println("n is od + " + j);
		 			// System.out.printf("%d %d %d %d\nABCD\n", a,b,c,d);
		 		}
		 	}
		 	if(m % 2 == 1){
		 		for(int i = 0; i < n-1; i += 2){
		 			int r = arr[i][m-1] + arr[i+1][m-1];
		 			int a = 0, b = arr[i][m-1], c = 0, d=arr[i+1][m-1];
		 			if(r == 1) print1(i+1,m-1, a,b,c,d);
		 			else if(r==2) print2(i+1,m-1, a,b,c,d);
		 			// System.out.println("m is odd + " + i);
		 		}	
		 	}

		 	if(n % 2 == 1 && m % 2 == 1){
		 		if(arr[n-1][m-1] == 1){
		 			print1(n-1, m-1,0,0,0,1);
		 			// System.out.println("ekeui de ");
		 		}
		 	}

		}
		
		
	}

	public static void print3(int i, int j, int a, int b, int c, int d){
		if(a == 0){
			System.out.printf("%d %d %d %d %d %d\n", i, j+1, i+1,j, i+1, j+1);//b c d
		}else if(b == 0){
			System.out.printf("%d %d %d %d %d %d\n", i,j,i+1,j,i+1,j+1); // a c d
		}else if(c == 0){
			System.out.printf("%d %d %d %d %d %d\n", i,j,i,j+1,i+1,j+1); //a b d
		}else{
			System.out.printf("%d %d %d %d %d %d\n", i,j,i+1,j,i,j+1); //a c b
		}
	}

	public static void print2(int i, int j, int a, int b, int c, int d){
		if(a == 0 && b==0){
			System.out.printf("%d %d %d %d %d %d\n", i,j, i+1,j, i,j+1); //a c b
			a = 1;
			b = 1;
			c = 0;
			print3(i, j, a, b, c, d);
		}else if(a == 0 && c == 0){
			System.out.printf("%d %d %d %d %d %d\n", i,j, i+1,j, i,j+1); //a c b
			a = 1;
			c = 1;
			b = 0;
			print3(i, j, a, b, c, d);
		}else if(a == 0 && d == 0){
			System.out.printf("%d %d %d %d %d %d\n", i,j, i+1,j+1, i,j+1); //a d b
			a = 1;
			d = 1;
			b = 0;
			print3(i, j, a, b, c, d);
		}else if(b == 0 && d == 0){
			System.out.printf("%d %d %d %d %d %d\n", i,j,i+1,j+1,i,j+1); //a d b
			a = 0;
			d = 1;
			b = 1;
			print3(i, j, a, b, c, d);
		}else if(c == 0 && d == 0){
			System.out.printf("%d %d %d %d %d %d\n", i+1,j,i+1,j+1,i,j+1); //c d b
			c = 1;
			d = 1;
			b = 0;
			print3(i, j, a, b, c, d);
		}else if(c == 0 && b == 0){
			System.out.printf("%d %d %d %d %d %d\n", i+1,j, i+1,j+1, i,j+1); //c d b
			c = 1;
			d = 0;
			b = 1;
			print3(i, j, a, b, c, d);
		}
	}

	public static void print1(int i, int j, int a, int b, int c, int d){
		if(a == 1){
			System.out.printf("%d %d %d %d %d %d\n", i, j, i+1,j, i+1, j+1);//a c d
			a = 0;
			c=1;
			d=1;
			print2(i, j,a,b,c,d);
		}else if(b == 1){
			System.out.printf("%d %d %d %d %d %d\n", i,j+1,i+1,j,i+1,j+1); //b c d
			b=0;
			c=1;
			d=1;
			print2(i, j,a,b,c,d);
		}else if(c == 1){
			System.out.printf("%d %d %d %d %d %d\n", i,j, i+1,j, i+1,j+1); // a c d
			c=0;
			a=1;
			d=1;
			print2(i, j,a,b,c,d);
		}else{
			System.out.printf("%d %d %d %d %d %d\n", i,j,i+1,j,i+1,j+1); // a c d
			d=0;
			a=1;
			c=1;
			print2(i, j,a,b,c,d);
		}
	}

	public static void print4(int i, int j, int a, int b, int c, int d){
		System.out.printf("%d %d %d %d %d %d\n", i,j,i+1,j,i+1,j+1); // a c d
		a=0;
		d=0;
		c=0;
		print1(i,j,a,b,c,d);
	}


	
}
